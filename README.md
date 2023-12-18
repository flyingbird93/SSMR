# SSMR
a novel Semantic and Style based Multiple Reference learning (SSMR) for artistic aesthetic image assessment

## Downloads
### [Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4585919), [Datasets](https://github.com/Dreemurr-T/BAID)

## Abstract
Artistic Image Aesthetic Assessment (AIAA) is an emerging paradigm that predicts the aesthetic score as the popular aesthetic taste for an artistic image. Previous AIAA takes a single image as input to predict the aesthetic score of the image. However, most existing AIAA methods fail dramatically to predict the artistic images with a large variance of artistic subjective voting with only a single image. People are good at employing multiple similar references for making relative comparisons. Motivated by the practice that people considers similar semantics and specific artistic style to keep the consistency of the voting result, we present a novel Semantic and Style based Multiple Reference learning (SSMR) to mimic this natural process. Our novelty is mainly two-fold: (a) Similar Reference Index Generation (SRIG) module that considers artistic attribution of semantics and style to generate the index of reference images; (b) Multiple Reference Graph Reasoning (MRGR) module that employs graph convolutional network (GCN) to initialize and reason by adjusting the weight of edges with intrinsic relationships among multiple images. Our evaluation with the benchmark BAID, VAPS and TAD66K datasets demonstrates that the proposed SSMR outperforms state-of-the-art AIAA methods.

## Framework
![微信图片_20231218152720](https://github.com/flyingbird93/SSMR/assets/16755407/90cf9090-3c37-42e0-b2b0-196417c053e8)


## Usage

### Requirements
Python3, requirements.txt

### Build
For pytorch 1.x:

    cd trilinear_cpp
    sh setup.sh

Please also replace the following lines:
```
# in image_adaptive_lut_train_paired.py, image_adaptive_lut_evaluation.py, demo_eval.py, and image_adaptive_lut_train_unpaired.py
from models import * --> from models_x import *
# in demo_eval.py
result = trilinear_(LUT, img) --> _, result = trilinear_(LUT, img)
# in image_adaptive_lut_train_paired.py and image_adaptive_lut_evaluation.py
combine_A = trilinear_(LUT,img) --> _, combine_A = trilinear_(LUT,img)
```

### Training
#### paired training
     python3 image_adaptive_lut_train_paired_with_cross_attention.py

### Evaluation
we provide the best model of MIT FIVEK dataset.[pretrain model](https://pan.baidu.com/s/1_cChj5afS0pxb39cCacEGA)密码1024

1. use python to generate and save the test images:

       python3 image_adaptive_lut_evaluation_with_cross_attention.py

speed can also be tested in above code.

2. use matlab to calculate the indexes used in our paper:

       average_psnr_ssim.m


### Tools
You can generate identity 3DLUT with arbitrary dimension by using `utils/generate_identity_3DLUT.py` as follows:

```
# you can replace 36 with any number you want
python3 utils/generate_identity_3DLUT.py -d 36
```


## Citation
```
@inproceedings{shirgb,
  title={RGB and LUT based Cross Attention Network for Image Enhancement},
  author={Shi, Tengfei and Chen, Chenglizhao and He, Yuanbo and Hao, Aimin},
  booktitle={34rd British Machine Vision Conference 2023, BMVC 2023, Aberdeen, UK, November 20-24},
  year={2023},
}

```

## Reference
Thanks for the code: [3D LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT)
