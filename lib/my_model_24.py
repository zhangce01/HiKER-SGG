from os import environ as os_environ
import sys
import pickle
import numpy as np
from torch import tensor as torch_tensor, float32 as torch_float32, zeros as torch_zeros, cat as torch_cat, from_numpy as torch_from_numpy, inverse as torch_inverse, cat as torch_cat, zeros as torch_zeros, LongTensor as torch_LongTensor, bool as torch_bool, log as torch_log, int64 as torch_int64
from torch.cuda import current_device
from torch.nn import Linear, Sequential, Module, AvgPool2d
from torch.nn.parallel import replicate, parallel_apply
from torch.nn.functional import cross_entropy as F_cross_entropy, softmax as F_softmax, kl_div as F_kl_div, log_softmax as F_log_softmax, nll_loss as F_nll_loss
from torch.nn.utils.rnn import PackedSequence
from torchvision.ops import nms, roi_align
from config import BATCHNORM_MOMENTUM
from lib.resnet import resnet_l4
from lib.fpn.box_utils import bbox_overlaps, center_size
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg
from lib.pytorch_misc import transpose_packed_sequence_inds, onehot_logits, arange, enumerate_by_image, diagonal_inds, Flattener
from lib.surgery import filter_dets
from lib.my_ggnn_10 import GGNN
from lib.my_util import adj_normalize


np.set_printoptions(threshold=sys.maxsize)

MODES = ('sgdet', 'sgcls', 'predcls')
CURRENT_DEVICE = current_device()


class GGNNRelReason(Module):
    """
    Module for relationship classification.
    """
    def __init__(self, graph_path, emb_path, mode='sgdet', num_obj_cls=151, num_rel_cls=51, obj_dim=4096, rel_dim=4096,
                time_step_num=3, hidden_dim=512, output_dim=512, use_knowledge=True, use_embedding=True, refine_obj_cls=False, with_clean_classifier=None, with_transfer=None, sa=None, config=None):

        super(GGNNRelReason, self).__init__()
        assert mode in MODES
        self.mode = mode
        self.with_clean_classifier = with_clean_classifier
        self.with_transfer = with_transfer
        self.num_obj_cls = num_obj_cls
        self.num_rel_cls = num_rel_cls
        self.obj_dim = obj_dim
        self.rel_dim = rel_dim


        self.obj_proj = Linear(self.obj_dim, hidden_dim)
        self.rel_proj = Linear(self.rel_dim, hidden_dim)

        assert not (refine_obj_cls and mode == 'predcls')

        self.ggnn = GGNN(time_step_num=time_step_num, hidden_dim=hidden_dim, output_dim=output_dim,
                         emb_path=emb_path, graph_path=graph_path, refine_obj_cls=refine_obj_cls,
                         use_knowledge=use_knowledge, use_embedding=use_embedding, config=config, with_clean_classifier=with_clean_classifier, with_transfer=with_transfer, sa=sa, num_obj_cls=num_obj_cls, num_rel_cls=num_rel_cls)


    def forward(self, im_inds, obj_fmaps, obj_logits, rel_inds, vr, obj_labels=None, boxes_per_cls=None):
        """
        Reason relationship classes using knowledge of object and relationship coccurrence.
        """

        # print(rel_inds.shape)
        # (num_rel, 3)
        if self.mode == 'predcls':
            #breakpoint()
            obj_logits = onehot_logits(obj_labels.data, self.num_obj_cls).clone().detach()
        obj_probs = F_softmax(obj_logits, 1)

        obj_fmaps = self.obj_proj(obj_fmaps)
        vr = self.rel_proj(vr)

        rel_logits = []
        obj_logits_refined = []
        scpred_softmax = []
        scent_softmax= []
        for (_, obj_s, obj_e), (_, rel_s, rel_e) in zip(enumerate_by_image(im_inds.data), enumerate_by_image(rel_inds[:,0])):
            #breakpoint()z
            rl, ol, scpred, scent = self.ggnn(rel_inds[rel_s:rel_e, 1:] - obj_s, obj_probs[obj_s:obj_e], obj_fmaps[obj_s:obj_e], vr[rel_s:rel_e])
            rel_logits.append(rl)
            obj_logits_refined.append(ol)
            scpred_softmax.append(scpred)
            scent_softmax.append(scent)

        rel_logits = torch_cat(rel_logits, 0)
        scpred_softmax = torch_cat(scpred_softmax, 0)

        if self.ggnn.refine_obj_cls:
            obj_logits_refined = torch_cat(obj_logits_refined, 0)
            obj_logits = obj_logits_refined
            scent_softmax = torch_cat(scent_softmax, 0)

        obj_probs = obj_logits
        if self.mode == 'sgdet' and not self.training:
            # NMS here for baseline
            nms_mask = obj_probs.data.clone()
            nms_mask.zero_()
            for c_i in range(1, obj_probs.size(1)):
                scores_ci = obj_probs.data[:, c_i]
                boxes_ci = boxes_per_cls.data[:, c_i]
                keep = nms(boxes=boxes_ci, scores=scores_ci, iou_threshold=0.3)
                # keep = apply_nms(scores_ci, boxes_ci,
                #                     pre_nms_topn=scores_ci.size(0), post_nms_topn=scores_ci.size(0),
                #                     nms_thresh=0.3)
                # print('my_model_24.GGNNRelReason.forward: keep.size() =', keep.size())
                num_out = min(len(keep), scores_ci.size(0))
                keep = keep[:num_out].long()

                nms_mask[:, c_i][keep] = 1

            obj_preds = torch_tensor(nms_mask * obj_probs.data, requires_grad=False, device=CURRENT_DEVICE, dtype=torch_float32)[:,1:].max(1)[1] + 1
        else:
            obj_preds = obj_labels if obj_labels is not None else obj_probs[:,1:].max(1)[1] + 1
            
        # print(rel_logits.shape, scpred.shape)

        return obj_logits, obj_preds, rel_logits, scpred_softmax, scent_softmax


class KERN(Module):
    """
    Knowledge-Embedded Routing Network
    """
    def __init__(self, classes, rel_classes, graph_path, emb_path, mode='sgdet', num_gpus=1,
                 require_overlap_det=True, pooling_dim=4096, use_resnet=False, thresh=0.01,
                 use_proposals=False,
                 ggnn_rel_time_step_num=3,
                 ggnn_rel_hidden_dim=512,
                 ggnn_rel_output_dim=512, use_knowledge=True, use_embedding=True, refine_obj_cls=False,
                 rel_counts_path=None, class_volume=1.0, with_clean_classifier=None, with_transfer=None, sa=None, config=None):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param require_overlap_det: Whether two objects must intersect
        """
        super(KERN, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        self.devices = [int(x) for x in os_environ['CUDA_VISIBLE_DEVICES'].split(',')]
        assert self.num_gpus == len(self.devices)
        assert mode in MODES
        self.mode = mode
        self.pooling_size = 7
        self.obj_dim = 2048 if use_resnet else 4096
        self.rel_dim = self.obj_dim
        self.pooling_dim = pooling_dim

        self.require_overlap = require_overlap_det and self.mode == 'sgdet'

        self.detector = ObjectDetector(
            classes=classes,
            mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64
        )


        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                              dim=1024 if use_resnet else 512)

        if use_resnet:
            self.roi_fmap = Sequential(
                resnet_l4(relu_end=False),
                AvgPool2d(self.pooling_size),
                Flattener(),
            )
        else:
            roi_fmap = [
                Flattener(),
                load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096, pretrained=False).classifier,
            ]
            if pooling_dim != 4096:
                roi_fmap.append(Linear(4096, pooling_dim))
            self.roi_fmap = Sequential(*roi_fmap)
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier

        self.ggnn_rel_reason = GGNNRelReason(mode=self.mode,
                                             num_obj_cls=len(self.classes),
                                             num_rel_cls=len(rel_classes),
                                             obj_dim=self.obj_dim,
                                             rel_dim=self.rel_dim,
                                             time_step_num=ggnn_rel_time_step_num,
                                             hidden_dim=ggnn_rel_hidden_dim,
                                             output_dim=ggnn_rel_output_dim,
                                             emb_path=emb_path,
                                             graph_path=graph_path,
                                             refine_obj_cls=refine_obj_cls,
                                             use_knowledge=use_knowledge,
                                             use_embedding=use_embedding,
                                             with_clean_classifier=with_clean_classifier,
                                             with_transfer=with_transfer,
                                             sa=sa,
                                             config=config,
                                             )

        if rel_counts_path is not None:
            with open(rel_counts_path, 'rb') as fin:
                rel_counts = pickle.load(fin)
            beta = (class_volume - 1.0) / class_volume
            self.rel_class_weights = (1.0 - beta) / (1 - (beta ** rel_counts))
            self.rel_class_weights *= float(self.num_rels) / np.sum(self.rel_class_weights)
        else:
            self.rel_class_weights = np.ones((self.num_rels,))

        self.rel_class_weights = torch_tensor(self.rel_class_weights, requires_grad=False, device=CURRENT_DEVICE, dtype=torch_float32)

        # self.with_clean_classifier = config.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER
        # self.with_transfer = config.MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER
        # self.sa = config.MODEL.ROI_RELATION_HEAD.SA


    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels

            if test:
            prob dists, boxes, img inds, maxscores, classes

        """
        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)
        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors

        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)


        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)
        rois = torch_cat((im_inds[:, None].float(), boxes), 1)

        result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)

        #breakpoint()
        vr = self.visual_rep(result.fmap.detach(), rois, rel_inds[:, 1:])
        # print(vr.shape)

        result.rm_obj_dists, result.obj_preds, result.rel_dists, result.scpred_softmax, result.scent_softmax = self.ggnn_rel_reason(
            im_inds=im_inds,
            obj_fmaps=result.obj_fmap,
            obj_logits=result.rm_obj_dists,
            vr=vr,
            rel_inds=rel_inds,
            obj_labels=result.rm_obj_labels if self.training or self.mode == 'predcls' else None,
            boxes_per_cls=result.boxes_all
        )

        if self.training:
            return result

        twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
        result.obj_scores = result.rm_obj_dists.view(-1)[twod_inds]

        # Bbox regression
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors

        rel_rep = result.rel_dists
        # rel_rep = F_softmax(result.rel_dists, dim=1)

        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:], rel_rep)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def visual_rep(self, features, rois, pair_inds):
        """
        Classify the features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds inds to use when predicting
        :return: score_pred, a [num_rois, num_classes] array
                 box_pred, a [num_rois, num_classes, 4] array
        """
        assert pair_inds.size(1) == 2
        return self.roi_fmap(self.union_boxes(features, rois, pair_inds))

    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        # Get the relationship candidates
        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)

                # if there are fewer then 100 things then we might as well add some?
                amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()

            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch_cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)

        return rel_inds


    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = roi_align(features, rois, output_size=[self.pooling_size, self.pooling_size], spatial_scale=1/16)
        # feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
        #     features, rois)
        # print('my_model_24.KERN.obj_feature_map: feature_pool.size() =', feature_pool.size())
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))


    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = replicate(self, devices=self.devices)
        outputs = parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])
        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs


    def obj_loss(self, result):
        if self.ggnn_rel_reason.ggnn.refine_obj_cls:
            return F_nll_loss(torch_log(result.rm_obj_dists + 1e-6), result.rm_obj_labels)
            # return F_cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
        else:
            return torch_zeros(1, requires_grad=False, device=CURRENT_DEVICE, dtype=torch_float32)

    def rel_loss(self, result):
        return F_nll_loss(torch_log(result.rel_dists + 1e-10), result.rel_labels[:, -1], weight=self.rel_class_weights)


    def scpred_loss(self, result):
        scpred_label = torch_zeros((result.scpred_softmax.shape[0]), requires_grad=False).type(torch_LongTensor).to(CURRENT_DEVICE)
        rel_labels = result.rel_labels[:, -1]
        doing = [14, 37, 47, 38]
        wear = [48, 49]
        superon = [28, 34, 35, 26, 24, 40, 41, 31, 18]
        superat = [29, 25, 6]
        position = [10, 33, 8, 4, 2, 13]
        superin = [15, 22, 12, 45, 46]
        superof = [16, 5, 50, 23, 32, 27, 36, 30]
        superto = [1, 7, 42, 9, 19, 17, 44]
        superother = [3, 11, 20, 21, 39, 43]
        superon1 = [28, 34, 35, 18]
        superon2 = [26, 24, 40, 41]
        superon3 = [31]
        superof1 = [16, 5, 50]
        superof2 = [23, 32]
        superof3 = [27, 36, 30]
        superto1 = [1, 7, 42, 9]
        superto2 = [19, 17, 44]
        scpred2_loss = torch_zeros(1, requires_grad=False, device=CURRENT_DEVICE, dtype=torch_float32)
        for i in range(result.scpred_softmax.shape[0]):
            if result.rel_labels[i, -1] == 0:
                scpred_label[i] = 0
            elif result.rel_labels[i, -1] in doing:
                scpred_label[i] = 1
            elif result.rel_labels[i, -1] in wear:
                scpred_label[i] = 2
            elif result.rel_labels[i, -1] in superon:
                scpred_label[i] = 3
                if result.rel_labels[i, -1] in superon1:
                    scpred2_label = torch_tensor([0], requires_grad=False, device=CURRENT_DEVICE, dtype=torch_int64)
                    scpred2_loss += F_nll_loss(torch_log(result.scpred_softmax[i][10:13].unsqueeze(0) + 1e-10), scpred2_label) / result.scpred_softmax.shape[0]
                elif result.rel_labels[i, -1] in superon2:
                    scpred2_label = torch_tensor([1], requires_grad=False, device=CURRENT_DEVICE, dtype=torch_int64)
                    scpred2_loss += F_nll_loss(torch_log(result.scpred_softmax[i][10:13].unsqueeze(0) + 1e-10), scpred2_label) / result.scpred_softmax.shape[0]
                elif result.rel_labels[i, -1] in superon3:
                    scpred2_label = torch_tensor([2], requires_grad=False, device=CURRENT_DEVICE, dtype=torch_int64)
                    scpred2_loss += F_nll_loss(torch_log(result.scpred_softmax[i][10:13].unsqueeze(0) + 1e-10), scpred2_label) / result.scpred_softmax.shape[0]
            elif result.rel_labels[i, -1] in superat:
                scpred_label[i] = 4
            elif result.rel_labels[i, -1] in position:
                scpred_label[i] = 5
            elif result.rel_labels[i, -1] in superin:
                scpred_label[i] = 6
            elif result.rel_labels[i, -1] in superof:
                scpred_label[i] = 7
                if result.rel_labels[i, -1] in superof1:
                    scpred2_label = torch_tensor([0], requires_grad=False, device=CURRENT_DEVICE, dtype=torch_int64)
                    scpred2_loss += F_nll_loss(torch_log(result.scpred_softmax[i][13:16].unsqueeze(0) + 1e-10), scpred2_label) / result.scpred_softmax.shape[0]
                elif result.rel_labels[i, -1] in superof2:
                    scpred2_label = torch_tensor([1], requires_grad=False, device=CURRENT_DEVICE, dtype=torch_int64)
                    scpred2_loss += F_nll_loss(torch_log(result.scpred_softmax[i][13:16].unsqueeze(0) + 1e-10), scpred2_label) / result.scpred_softmax.shape[0]
                elif result.rel_labels[i, -1] in superof3:
                    scpred2_label = torch_tensor([2], requires_grad=False, device=CURRENT_DEVICE, dtype=torch_int64)
                    scpred2_loss += F_nll_loss(torch_log(result.scpred_softmax[i][13:16].unsqueeze(0) + 1e-10), scpred2_label) / result.scpred_softmax.shape[0]
            elif result.rel_labels[i, -1] in superto:
                scpred_label[i] = 8
                if result.rel_labels[i, -1] in superto1:
                    scpred2_label = torch_tensor([0], requires_grad=False, device=CURRENT_DEVICE, dtype=torch_int64)
                    scpred2_loss += F_nll_loss(torch_log(result.scpred_softmax[i][16:18].unsqueeze(0) + 1e-10), scpred2_label) / result.scpred_softmax.shape[0]
                elif result.rel_labels[i, -1] in superto2:
                    scpred2_label = torch_tensor([1], requires_grad=False, device=CURRENT_DEVICE, dtype=torch_int64)
                    scpred2_loss += F_nll_loss(torch_log(result.scpred_softmax[i][16:18].unsqueeze(0) + 1e-10), scpred2_label) / result.scpred_softmax.shape[0]
            elif result.rel_labels[i, -1] in superother:
                scpred_label[i] = 9
        scpred_loss = F_nll_loss(torch_log(result.scpred_softmax[:, 0:10] + 1e-10), scpred_label)
        # print(result.scpred_softmax[:, 0:10].shape, scpred_label.shape)
        # print(scpred_loss, scpred2_loss)
        return scpred_loss + scpred2_loss

    def scent_loss(self, result):
        if self.ggnn_rel_reason.ggnn.refine_obj_cls:
            scent_label = torch_zeros((result.scent_softmax.shape[0]), requires_grad=False).type(torch_LongTensor).to(CURRENT_DEVICE)
            obj_labels = result.rm_obj_labels
            """
            part = [3, 40, 43, 44, 46, 58, 59, 61, 74, 82, 83, 84, 127, 129, 130, 6, 144, 57, 85]
            artifact = [4, 15, 17, 18, 19, 25, 34, 42, 50, 54, 62, 71, 75, 77, 88, 92, 97, 99, 100, 101, 102, 107, 132, 146, 148, 10, 140, 30, 47, 69, 72, 116, 117, 118, 123, 125]
            person = [20, 29, 53, 56, 68, 70, 78, 79, 90, 91, 149, 98, 119]
            clothes = [16, 31, 55, 60, 66, 67, 87, 111, 112, 113, 120, 122, 128]
            vehicle = [1, 11, 14, 23, 26, 80, 95, 135, 137, 142]
            flora = [21, 48, 51, 73, 96, 141]
            location = [7, 81, 114, 124, 131, 143, 121]
            furniture = [9, 28, 32, 36, 38, 39, 93, 108, 110, 126, 35]
            animal = [2, 8, 27, 33, 37, 41, 52, 64, 89, 109, 150, 12]
            structure = [13, 45, 63, 76, 103, 104, 105, 115, 133, 134, 136, 138, 139, 145, 147]
            building = [22, 24, 65, 106]
            food = [5, 49, 86, 94]
            """
            for i in range(result.scent_softmax.shape[0]):
                if result.rm_obj_labels[i] == 0:
                    scent_label[i] = 0
                elif result.rm_obj_labels[i] in [3, 40, 43, 44, 46, 58, 59, 61, 74, 82, 83, 84, 127, 129, 130, 6, 144, 57, 85]:
                    scent_label[i] = 1
                elif result.rm_obj_labels[i] in [4, 15, 17, 18, 19, 25, 34, 42, 50, 54, 62, 71, 75, 77, 88, 92, 97, 99, 100, 101, 102, 107, 132, 146, 148, 10, 140, 30, 47, 69, 72, 116, 117, 118, 123, 125]:
                    scent_label[i] = 2
                elif result.rm_obj_labels[i] in [20, 29, 53, 56, 68, 70, 78, 79, 90, 91, 149, 98, 119]:
                    scent_label[i] = 3
                elif result.rm_obj_labels[i] in [16, 31, 55, 60, 66, 67, 87, 111, 112, 113, 120, 122, 128]:
                    scent_label[i] = 4
                elif result.rm_obj_labels[i] in [1, 11, 14, 23, 26, 80, 95, 135, 137, 142]:
                    scent_label[i] = 5
                elif result.rm_obj_labels[i] in [21, 48, 51, 73, 96, 141]:
                    scent_label[i] = 6
                elif result.rm_obj_labels[i] in [7, 81, 114, 124, 131, 143, 121]:
                    scent_label[i] = 7
                elif result.rm_obj_labels[i] in [9, 28, 32, 36, 38, 39, 93, 108, 110, 126, 35]:
                    scent_label[i] = 8
                elif result.rm_obj_labels[i] in [2, 8, 27, 33, 37, 41, 52, 64, 89, 109, 150, 12]:
                    scent_label[i] = 9
                elif result.rm_obj_labels[i] in [13, 45, 63, 76, 103, 104, 105, 115, 133, 134, 136, 138, 139, 145, 147]:
                    scent_label[i] = 10
                elif result.rm_obj_labels[i] in [22, 24, 65, 106]:
                    scent_label[i] = 11
                elif result.rm_obj_labels[i] in [5, 49, 86, 94]:
                    scent_label[i] = 12
            return F_nll_loss(torch_log(result.scent_softmax + 1e-6), scent_label)
        else:
            return torch_zeros(1, requires_grad=False, device=CURRENT_DEVICE, dtype=torch_float32)

