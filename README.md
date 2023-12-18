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

### Training and Test
#### Training
- stage 1 Create Retrieval Index
```python
python index/create_semantic_style_feature_similar_refer_index.py
```
- stage 2 Extract SwinTransformer feature
```python
python index/full_size_feature_extract.py
```
- stage 3 Training GCN model
```python
python SSMR_AIAA.py --train True
```
### Evaluation
     python SSMR_AIAA.py --train False --test True
     
we provide the best model of BAID dataset.
[pretrain model](https://pan.baidu.com/s/1UJXOp8O9R24lfSed8vAyhw)密码1024



## Citation
```
@article{shisemantic,
  title={Semantic and Style Based Multiple Reference Learning for Artistic Image Aesthetic Assessment},
  author={Shi, Tengfei and Li, Xuan and Hao, Aimin and others},
  journal={NeuroComputing}
}

```

