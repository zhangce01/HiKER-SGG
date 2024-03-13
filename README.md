# Graph Bridging Network (GB-Net)
Code for the ECCV 2020 paper: [Bridging Knowledge Graphs to Generate Scene Graphs](https://arxiv.org/pdf/2001.02314.pdf)
```
@InProceedings{Zareian_2020_ECCV,
author = {Zareian, Alireza and Karaman, Svebor and Chang, Shih-Fu},
title = {Bridging Knowledge Graphs to Generate Scene Graphs},
booktitle = {Proceedings of the European conference on computer vision (ECCV)},
month = {August},
year = {2020}
}
```

Instructions to reproduce all numbers in table 1 and table 2 of our paper:

First, download and unpack Visual Genome images: [part 2](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) and [part 2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

Extract these two zip files and put the images in the same folder.

Then download VG metadata preprocessed by \[37\]: [annotations](http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG.h5), [class info](http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG-dicts.json),and [image metadata](http://svl.stanford.edu/projects/scene-graph/VG/image_data.json)

Copy those three files in a single folder

Then update `config.py` to with a path to the aforementioned data, as well as the absolute path to this directory.

Now download the pretrained faster r-cnn checkpoint trained by [42] from https://www.dropbox.com/s/cfyqhskypu7tp0q/vg-24.tar?dl=0 and place in `checkpoints/vgdet`

The next step is to configure a python environment and install pytorch. To do that, first make sure CUDA 9 is installed, and then download https://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl and pip install the downloaded `whl` file. Then install the rest of required packages by running `pip install -r requirements.txt`. This includes jupyter, as you need it to run the notebooks.

Finally, run the following to produce numbers for each table (In some cases order matters):
```
Table 1, Column 8, Rows 17-24: train: ipynb/train_predcls/0045.ipynb, evaluate: ipynb/eval_predcls/0011.ipynb
Table 1, Column 8, Rows 9-16: train: ipynb/train_sgcls/0051.ipynb, evaluate: ipynb/eval_sgcls/0015.ipynb
Table 1, Column 8, Rows 1-8: train: ipynb/train_predcls/0132.ipynb, evaluate: ipynb/eval_sgdet/0027.ipynb

Table 1, Column 9, Rows 17-24: train: ipynb/train_predcls/0135.ipynb, evaluate: ipynb/eval_predcls/0025.ipynb
Table 1, Column 9, Rows 9-16: train: ipynb/train_sgcls/0145.ipynb, evaluate: ipynb/eval_sgcls/0039.ipynb
Table 1, Column 9, Rows 1-8: train: ipynb/train_predcls/0135.ipynb, evaluate: ipynb/eval_sgdet/0035.ipynb

Table 2, Row 1, Columns 6-9: train: ipynb/train_predcls/0140.ipynb, evaluate: ipynb/eval_predcls/0030.ipynb
Table 2, Row 1, Columns 2-5: train: ipynb/train_predcls/0140.ipynb, evaluate: ipynb/eval_sgdet/0028.ipynb

Table 2, Row 2, Columns 6-9: train: ipynb/train_predcls/0134.ipynb, evaluate: ipynb/eval_predcls/0024.ipynb
Table 2, Row 2, Columns 2-5: train: ipynb/train_predcls/0134.ipynb, evaluate: ipynb/eval_sgdet/0034.ipynb

Table 2, Row 3, Columns 6-9: train: ipynb/train_predcls/0136.ipynb, evaluate: ipynb/eval_predcls/0026.ipynb
Table 2, Row 3, Columns 2-5: train: ipynb/train_predcls/0136.ipynb, evaluate: ipynb/eval_sgdet/0036.ipynb

Table 2, Row 4, Columns 6-9: train: ipynb/train_predcls/0132.ipynb, evaluate: ipynb/eval_predcls/0022.ipynb
Table 2, Row 4, Columns 2-5: train: ipynb/train_predcls/0132.ipynb, evaluate: ipynb/eval_sgdet/0027.ipynb
```

Moreover, SGCls results for table 2, which is missing from the paper due to space constraint, can be produced by:
```
Row 1: train: ipynb/train_predcls/0150.ipynb, evaluate: ipynb/eval_predcls/0041.ipynb
Row 2: train: ipynb/train_predcls/0144.ipynb, evaluate: ipynb/eval_predcls/0038.ipynb
Row 3: train: ipynb/train_predcls/0146.ipynb, evaluate: ipynb/eval_predcls/0040.ipynb
Row 4: train: ipynb/train_predcls/0142.ipynb, evaluate: ipynb/eval_predcls/0037.ipynb
```

To skip training, you may download all our pretrained checkpoints from [here](https://www.dropbox.com/sh/r62mzgsg1f81776/AAAQKzPD8qJrBYeYzNHJ0p5Xa?dl=0) and place in the `checkpoints/` folder. Then you only need to run notebooks in `ipynb/eval_...`

If GPU is not available, to skip deploying the model altogether, you may download our pre-computed model outputs from [here](https://www.dropbox.com/sh/rbnkcnfh0bmw08m/AACVBegZ14YGG9XwcsmJFxFua?dl=0) and place in the `caches/` folder. Then if you run any notebook in `ipynb/eval_...`, it automatically uses the cached results and does not deploy the model. Note that there is no need to run the cell that creates the model (`detector = ...`) as well as the next one that transfers it to cuda (`detector.cuda()`) and the next one that loads the checkpoint (`ckpt = ...`). Only run the rest of the cells.

Finally, to avoid running the code, you may just open the notebooks in `ipynb/eval_...` and scroll down to see the evaluation results.

Note if you get cuda-related errors, it might be due to the cuda compatibility options that were used to compile this library. In that case, you need to change the compatibility in `lib/fpn/nms/src/cuda/Makefile` and `lib/fpn/roi_align/src/cuda/Makefile` and rebuild both by running make clean and then make in both directories. 
Also note that pytorch 0.3.0 only has pre-built binaries for up to cuda 9. In order to run this with cuda 10 and newer GPUs, you need to build pytorch from source.

Acknowledgement: This repository is based on our references [\[1\]](https://github.com/yuweihao/KERN) and [\[42\]](https://github.com/rowanz/neural-motifs)

[1] Chen, Tianshui, et al. "Knowledge-Embedded Routing Network for Scene Graph Generation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.

[37] Xu, Danfei, et al. "Scene graph generation by iterative message passing." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.

[42] Zellers, Rowan, et al. "Neural motifs: Scene graph parsing with global context." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

Created and maintained by [Alireza Zareian](https://www.linkedin.com/in/az2407/) at [DVMM](http://www.ee.columbia.edu/ln/dvmm/) - Columbia University.
