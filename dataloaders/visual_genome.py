"""
File that involves dataloaders for the Visual Genome dataset.
"""
import json
import os
import sys
import numpy as np
from os import environ as os_environ
from collections import defaultdict
from h5py import File as h5py_File
from os.path import join as os_path_join, exists as os_path_exists
from json import load as json_load
from numpy import array as np_array, where as np_where, \
    zeros_like as np_zeros_like, all as np_all, column_stack as np_column_stack, \
    zeros as np_zeros, int32 as np_int32
from numpy.random import random as np_random_random, choice as np_random_choice
from PIL import Image
from PIL.Image import open as Image_open, FLIP_LEFT_RIGHT as Image_FLIP_LEFT_RIGHT, LANCZOS as Image_LANCZOS
from torch.utils.data import Dataset, DataLoader
from torch import save as torch_save, load as torch_load
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from pycocotools.coco import COCO
from dataloaders.blob import Blob
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from config import VG_IMAGES, IM_DATA_FN, VG_SGG_FN, VG_SGG_DICT_FN, BOX_SCALE, IM_SCALE, PROPOSAL_FN
from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, \
    RandomOrder, Hue, random_crop
from PIL import ImageDraw
from .corruptions import gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, zoom_blur, motion_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression, speckle_noise, gaussian_blur, spatter, saturate, sunglare, waterdrop, wildfire_smoke, rain, dust
import matplotlib.pyplot as plt


PRINTING = int(os_environ.get('printing', False))


class VG(Dataset):
    def __init__(self, mode, roidb_file=VG_SGG_FN, dict_file=VG_SGG_DICT_FN,
                 image_file=IM_DATA_FN, filter_empty_rels=True, num_im=-1, num_val_im=5000,
                 filter_duplicate_rels=True, filter_non_overlap=True,
                 use_proposals=False, with_clean_classifier=None, get_state=None, caching=False, use_cache=False, test_n=False):
        """
        Torch dataset for VisualGenome
        :param mode: Must be train, test, or val
        :param roidb_file:  HDF5 containing the GT boxes, classes, and relationships
        :param dict_file: JSON Contains mapping of classes/relationships to words
        :param image_file: HDF5 containing image filenames
        :param filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
        :param filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
        :param num_im: Number of images in the entire dataset. -1 for all images.
        :param num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        :param proposal_file: If None, we don't provide proposals. Otherwise file for where we get RPN
            proposals
        """
        self.use_cache = use_cache
        self.caching = caching
        if mode not in ('test', 'train', 'val'):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format(mode))
        self.mode = mode
        self.test_n = test_n

        # Initialize
        self.roidb_file = roidb_file
        self.dict_file = dict_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap
        self.filter_duplicate_rels = filter_duplicate_rels and self.mode == 'train'

        self.ind_to_classes, self.ind_to_predicates = load_info(dict_file)  # contiguous 151, 51 containing __background__
        self.split_mask, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
            self.roidb_file, self.mode, num_im, num_val_im=num_val_im,
            filter_empty_rels=filter_empty_rels,
            filter_non_overlap=self.filter_non_overlap and self.is_train,
            dict_file=dict_file,
            with_clean_classifier=with_clean_classifier,
            get_state=get_state,
            ind_to_predicates=self.ind_to_predicates,
        )

        self.filenames = load_image_filenames(image_file)
        self.filenames = [self.filenames[i] for i in np_where(self.split_mask)[0]]

        if use_proposals:
            print("Loading proposals", flush=True)
            with h5py_File(PROPOSAL_FN, 'r') as p_h5:
                rpn_rois = p_h5['rpn_rois']
                rpn_scores = p_h5['rpn_scores']
                rpn_im_to_roi_idx = np_array(p_h5['im_to_roi_idx'][self.split_mask])
                rpn_num_rois = np_array(p_h5['num_rois'][self.split_mask])

            self.rpn_rois = []
            for i in range(len(self.filenames)):
                rpn_i = np_column_stack((
                    rpn_scores[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]],
                    rpn_rois[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]],
                ))
                self.rpn_rois.append(rpn_i)
        else:
            self.rpn_rois = None

        # You could add data augmentation here. But we didn't.
        # tform = []
        # if self.is_train:
        #     tform.append(RandomOrder([
        #         Grayscale(),
        #         Brightness(),
        #         Contrast(),
        #         Sharpness(),
        #         Hue(),
        #     ]))

        tform = [
            SquarePad(),
            Resize(IM_SCALE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform_pipeline = Compose(tform)

    @property
    def coco(self):
        """
        :return: a Coco-like object that we can use to evaluate detection!
        """
        anns = []
        for i, (cls_array, box_array) in enumerate(zip(self.gt_classes, self.gt_boxes)):
            for cls, box in zip(cls_array.tolist(), box_array.tolist()):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': i,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'ayy lmao'},
            'images': [{'id': i} for i in range(self.__len__())],
            'categories': [{'supercategory': 'person',
                               'id': i, 'name': name} for i, name in enumerate(self.ind_to_classes) if name != '__background__'],
            'annotations': anns,
        }
        fauxcoco.createIndex()
        return fauxcoco

    @property
    def is_train(self):
        return self.mode.startswith('train')

    @classmethod
    def splits(cls, *args, **kwargs):
        """ Helper method to generate splits of the dataset"""
        train = cls('train', *args, **kwargs)
        val = cls('val', *args, **kwargs)
        test = cls('test', *args, **kwargs)
        return train, val, test

    def __getitem__(self, index):
        fname = self.filenames[index]
        cache_path = os_path_join(f'cached_{self.mode}', f'{fname}.pt')
        if self.caching is True and (self.use_cache is True or os_path_exists(cache_path)):
            return torch_load(cache_path)

        image_unpadded = Image_open(fname).convert('RGB')
        w, h = image_unpadded.size
        max_side = max(w, h)

        if self.test_n:
            ################### Apply corruptions to the image ####################
            # image_unpadded = gaussian_noise(image_unpadded, severity=5)
            # image_unpadded = shot_noise(image_unpadded, severity=5)
            # image_unpadded = impulse_noise(image_unpadded, severity=5)
            # image_unpadded = defocus_blur(image_unpadded, severity=5)
            # image_unpadded = glass_blur(image_unpadded, severity=5)
            # image_unpadded = motion_blur(image_unpadded, severity=5)
            # image_unpadded = zoom_blur(image_unpadded, severity=5)
            # image_unpadded = snow(image_unpadded, severity=5)
            # image_unpadded = frost(image_unpadded, severity=5)
            # image_unpadded = fog(image_unpadded, severity=5)
            # image_unpadded = brightness(image_unpadded, severity=5)
            # image_unpadded = contrast(image_unpadded, severity=5)
            # image_unpadded = jpeg_compression(image_unpadded, severity=5)
            # image_unpadded = pixelate(image_unpadded, severity=5)
            # image_unpadded = elastic_transform(image_unpadded, severity=5)
            # image_unpadded = sunglare(image_unpadded, severity=5)
            # image_unpadded = waterdrop(image_unpadded, severity=5)
            # image_unpadded = wildfire_smoke(image_unpadded, severity=5)
            # image_unpadded = rain(image_unpadded, severity=5)
            # image_unpadded = dust(image_unpadded, severity=5)

            image_unpadded = Image.fromarray(image_unpadded.astype(np.uint8))

            # For debugging
            # print(image_unpadded.size)
            # display(image_unpadded)
            # sys.exit()

            ################### To visualize the corrupted images ###################
            # if index == 90:
            #     correpted_list = []
            #     image_unpadded1 = gaussian_noise(image_unpadded, severity=3)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = shot_noise(image_unpadded, severity=3)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = impulse_noise(image_unpadded, severity=3)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = defocus_blur(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = glass_blur(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = motion_blur(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = zoom_blur(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = snow(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = frost(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = fog(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = brightness(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = contrast(image_unpadded, severity=1)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = elastic_transform(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = pixelate(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = jpeg_compression(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = sunglare(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = waterdrop(image_unpadded, severity=3)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = wildfire_smoke(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = rain(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))
            #     image_unpadded1 = dust(image_unpadded, severity=5)
            #     correpted_list.append(Image.fromarray(image_unpadded1.astype(np.uint8)))

            #     # Show all the corrupted images in a single 4x5 plot
            #     fig, axs = plt.subplots(4, 5, figsize=(20, 12))
            #     axs = axs.flatten()
            #     name = ['Gaussian Noise', 'Shot Noise', 'Impulse Noise', 'Defocus Blur', 'Glass Blur', 'Motion Blur', 'Zoom Blur', 'Snow', 'Frost', 'Fog', 'Brightness', 'Contrast', 'Elastic Transform', 'Pixelate', 'JPEG Compression', 'Sunlight glare', 'Waterdrop', 'Wildfire Smoke', 'Rain', 'Dust']
            #     for img, ax in zip(correpted_list, axs):
            #         ax.imshow(img)
            #         ax.axis('off')
            #         ax.set_title(name[correpted_list.index(img)], fontsize=14)
            #     plt.tight_layout()
            #     plt.show()
            #     sys.exit()
            ##########################################################################

        # Optionally flip the image if we're doing training
        flipped = self.is_train and np_random_random() > 0.5
        gt_boxes = self.gt_boxes[index].copy()

        box_scale_factor = BOX_SCALE / max_side
        # Boxes are already at BOX_SCALE
        if self.is_train:

            # crop boxes that are too large. This seems to be only a problem for image heights, but whatevs
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clip(
                None, box_scale_factor * h)
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clip(
                None, box_scale_factor * w)

            # # crop the image for data augmentation
            # image_unpadded, gt_boxes = random_crop(image_unpadded, gt_boxes, BOX_SCALE, round_boxes=True)

        if flipped:
            scaled_w = int(box_scale_factor * float(w))
            # print("Scaled w is {}".format(scaled_w))
            image_unpadded = image_unpadded.transpose(Image_FLIP_LEFT_RIGHT)
            gt_boxes[:, [0, 2]] = scaled_w - gt_boxes[:, [2, 0]]

        if PRINTING: print(f'visual_genome: before: (w, h) = {(w, h)}')
        img_scale_factor = IM_SCALE / max_side
        if h > w:
            im_size = (IM_SCALE, int(w * img_scale_factor), img_scale_factor)
        elif h < w:
            im_size = (int(h * img_scale_factor), IM_SCALE, img_scale_factor)
        else:
            im_size = (IM_SCALE, IM_SCALE, img_scale_factor)

        if PRINTING: print(f'visual_genome: after: im_size = {im_size}')
        gt_rels = self.relationships[index].copy()
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.mode == 'train'
            old_size = gt_rels.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in gt_rels:
                all_rel_sets[(o0, o1)].append(r)
            gt_rels = [(k[0], k[1], np_random_choice(v)) for k,v in all_rel_sets.items()]
            gt_rels = np_array(gt_rels)

        entry = {
            'img': self.transform_pipeline(image_unpadded),
            'img_size': im_size,
            'gt_boxes': gt_boxes,
            'gt_classes': self.gt_classes[index].copy(),
            'gt_relations': gt_rels,
            'scale': IM_SCALE / BOX_SCALE,  # Multiply the boxes by this.
            'index': index,
            'flipped': flipped,
            'fn': fname,
        }

        if self.rpn_rois is not None:
            entry['proposals'] = self.rpn_rois[index]

        assertion_checks(entry)
        if self.caching is True: torch_save(entry, cache_path)
        # self.cached_data.append(entry)
        return entry

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def __len__(self):
        return len(self.filenames)

    @property
    def num_predicates(self):
        return len(self.ind_to_predicates)

    @property
    def num_classes(self):
        return len(self.ind_to_classes)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MISC. HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def assertion_checks(entry):
    im_size = tuple(entry['img'].size())
    if len(im_size) != 3:
        raise ValueError("Img must be dim-3")

    c, h, w = entry['img'].size()
    if c != 3:
        raise ValueError("Must have 3 color channels")

    num_gt = entry['gt_boxes'].shape[0]
    if entry['gt_classes'].shape[0] != num_gt:
        raise ValueError("GT classes and GT boxes must have same number of examples")

    assert (entry['gt_boxes'][:, 2] >= entry['gt_boxes'][:, 0]).all()
    assert (entry['gt_boxes'] >= -1).all()


def load_image_filenames(image_file, image_dir=VG_IMAGES):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    :param image_file: JSON file. Elements contain the param "image_id".
    :param image_dir: directory where the VisualGenome images are located
    :return: List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json_load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os_path_join(image_dir, basename)
        if os_path_exists(filename):
            fns.append(filename)
    assert len(fns) == 108073
    return fns


def load_graphs(graphs_file, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True,
                filter_non_overlap=False, dict_file=None, with_clean_classifier=None, ind_to_predicates=None, get_state=None):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    :param graphs_file: HDF5
    :param mode: (train, val, or test)
    :param num_im: Number of images we want
    :param num_val_im: Number of validation images
    :param filter_empty_rels: (will be filtered otherwise.)
    :param filter_non_overlap: If training, filter images that dont overlap.
    :return: image_index: numpy array corresponding to the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    with h5py_File(graphs_file, 'r') as roi_h5:
        data_split = roi_h5['split'][:]
        split = 2 if mode == 'test' else 0
        split_mask = data_split == split

        # Filter out images without bounding boxes
        split_mask &= roi_h5['img_to_first_box'][:] >= 0
        if filter_empty_rels:
            split_mask &= roi_h5['img_to_first_rel'][:] >= 0

        image_index = np_where(split_mask)[0]
        if num_im > -1:
            image_index = image_index[:num_im]
        if num_val_im > 0:
            if mode == 'val':
                image_index = image_index[:num_val_im]
            elif mode == 'train':
                image_index = image_index[num_val_im:]


        split_mask = np_zeros_like(data_split, dtype=bool)
        split_mask[image_index] = True

        # Get box information
        all_labels = roi_h5['labels'][:, 0]
        all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # will index later
        assert np_all(all_boxes[:, :2] >= 0)  # sanity check
        assert np_all(all_boxes[:, 2:] > 0)  # no empty box

        # convert from xc, yc, w, h to x1, y1, x2, y2
        all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
        all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

        im_to_first_box = roi_h5['img_to_first_box'][split_mask]
        im_to_last_box = roi_h5['img_to_last_box'][split_mask]
        im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
        im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

        # load relation labels
        _relations = roi_h5['relationships'][:]
        _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    # gt_attributes = []
    relationships = []
    pred_topk = []
    pred_num = 15
    pred_count=0
    # with open('./datasets/vg/VG-SGG-dicts-with-attri-info.json','r') as f:
    with open(dict_file,'r') as f:
        vg_dict_info = json_load(f)

    predicates_tree = vg_dict_info['predicate_count']
    #predicates_tree = json.load(open('./datasets/vg/predicate_wikipedia_count.json', 'r'))
    predicates_sort = sorted(predicates_tree.items(), key=lambda x:x[1], reverse=True)
    for pred_i in predicates_sort:
        if pred_count >= pred_num:
            break
        pred_topk.append(str(pred_i[0]))
        pred_count += 1

    if with_clean_classifier:
        print('Dataloader using BPL')
        root_classes = pred_topk
    else:
        print('Dataloader NOT using BPL')
        root_classes = None
    if get_state:
        root_classes = None
    root_classes_count = {}
    leaf_classes_count = {}
    all_classes_count = {}
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]
        # gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start: i_rel_end + 1]
            obj_idx = _relations[i_rel_start: i_rel_end + 1] - i_obj_start  # range is [0, num_box)
            assert np_all(obj_idx >= 0)
            assert np_all(obj_idx < boxes_i.shape[0])
            rels = np_column_stack((obj_idx, predicates))  # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np_zeros((0, 3), dtype=np_int32)

        if filter_non_overlap:
            assert mode == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np_where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue
        if root_classes is not None and mode == 'train':
            # print('old boxes: ', boxes_i)
            # print('old gt_classes_i: ', gt_classes_i)
            # print('old rels: ', rels)
            rel_temp = []
            boxmap_old2new = {}
            box_num = 0
            retain_box = []
            # print('rels: ',rels)
            for rel_i in rels:
                rel_i_pred = ind_to_predicates[rel_i[2]]
                if rel_i_pred not in all_classes_count:
                    all_classes_count[rel_i_pred] = 0
                all_classes_count[rel_i_pred] = all_classes_count[rel_i_pred] + 1
                if rel_i_pred not in root_classes or rel_i[2] == 0:
                    rel_i_leaf = rel_i

                    # if rel_i[0] not in boxmap_old2new:
                    # boxmap_old2new[rel_i[0]] = box_num
                    # retain_box.append(rel_i[0])
                    # box_num = box_num + 1
                    # if rel_i[1] not in boxmap_old2new:
                    # boxmap_old2new[rel_i[1]] = box_num
                    # retain_box.append(rel_i[1])
                    # box_num = box_num + 1
                    # rel_i_new[0] = boxmap_old2new[rel_i[0]]
                    # rel_i_new[1] = boxmap_old2new[rel_i[1]]
                    if rel_i_pred not in leaf_classes_count:
                        leaf_classes_count[rel_i_pred] = 0
                    leaf_classes_count[rel_i_pred] = leaf_classes_count[rel_i_pred] + 1
                    rel_temp.append(rel_i_leaf)
                if rel_i_pred in root_classes:
                    rel_i_root = rel_i
                    if rel_i_pred not in root_classes_count:
                        root_classes_count[rel_i_pred] = 0
                    if root_classes_count[rel_i_pred] < 1000: # Adjust the intensity of BPL here
                        rel_temp.append(rel_i_root)
                        root_classes_count[rel_i_pred] = root_classes_count[rel_i_pred] + 1
            if len(rel_temp) == 0:
                split_mask[image_index[i]] = 0
                continue
            else:
                rels = np_array(rel_temp, dtype=np_int32)

            # retain_box = np.array(retain_box, dtype=np.int64)
            # boxes_i = boxes_i[retain_box]
            # gt_classes_i = gt_classes_i[retain_box]
            # gt_attributes_i = gt_attributes_i[retain_box]

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        # gt_attributes.append(gt_attributes_i)
        relationships.append(rels)
    if PRINTING: print('mode: ',mode)
    if PRINTING: print('root_classes_count: ', root_classes_count)
    count_list = [0,]
    for i in root_classes_count:
        count_list.append(root_classes_count[i])
    if PRINTING: print('mean root class number: ', np_array(count_list).mean())
    if PRINTING: print('sum root class number: ', np_array(count_list).sum())

    if PRINTING: print('leaf_classes_count: ', leaf_classes_count)
    count_list = [0,]
    for i in leaf_classes_count:
        count_list.append(leaf_classes_count[i])
    if PRINTING: print('mean leaf class number: ', np_array(count_list).mean())
    if PRINTING: print('sum leaf class number: ', np_array(count_list).sum())
    # clean_classes_count = {}
    # clean_classes_count = root_classes_count.copy()
    # clean_classes_count.update(leaf_classes_count)
    # with open("./misc/clean_classes_count.json", "w") as dump_f:
    #     print('save clean_classes_count')
    #     json.dump(clean_classes_count, dump_f)
    if PRINTING: print('all_classes_count: ', all_classes_count)
    count_list = [0,]
    for i in all_classes_count:
        count_list.append(all_classes_count[i])
    # if split == 'train':
    #     with open("./misc/all_predicate_count.json", "w") as dump_f:
    #         print('save all_classes_count')
    #         json.dump(all_classes_count, dump_f)
    #     os._exit(0)
    if PRINTING: print('mean all class number: ', np_array(count_list).mean())
    if PRINTING: print('sum all class number: ', np_array(count_list).sum())
    if PRINTING: print('number images: ', split_mask.sum())

    return split_mask, boxes, gt_classes, relationships


def load_info(info_file):
    """
    Loads the file containing the visual genome label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
    """
    with open(info_file, 'r') as f:
        info = json_load(f)
    info['label_to_idx']['__background__'] = 0
    info['predicate_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return ind_to_classes, ind_to_predicates


def vg_collate(data, num_gpus=3, is_train=False, mode='det'):
    assert mode in ('det', 'rel')
    blob = Blob(mode=mode, is_train=is_train, num_gpus=num_gpus,
                batch_size_per_gpu=len(data) // num_gpus)
    for d in data:
        blob.append(d)
    blob.reduce()
    return blob


class VGDataLoader(DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def splits(cls, train_data, val_data, batch_size=3, num_workers=1, num_gpus=3, mode='det',
               **kwargs):
        assert mode in ('det', 'rel')
        train_load = cls(
            dataset=train_data,
            batch_size=batch_size * num_gpus,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=True),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        val_load = cls(
            dataset=val_data,
            batch_size=batch_size * num_gpus if mode=='det' else num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=False),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        return train_load, val_load
