# Sequence Generation Model for Multi-label Classification & pytorch0.4.1
- This is the code for our paper *SGM: Sequence Generation Model for Multi-label Classification* [[pdf]](https://arxiv.org/abs/1806.04822)
- Be careful: the provided code is based on the RCV1-V2 dataset. If you need to run the code on other datasets, please correspondly modify all program statements that relate to the specific name of the dataset.

***********************************************************

## Datasets
* RCV1-V2
* AAPD

Two datasets are available at https://drive.google.com/file/d/18-JOCIj9v5bZCrn9CIsk23W4wyhroCp_/view?usp=sharing

***************************************************************

## Requirements
* Ubuntu 18.0.4
* Python 3.7
* Pytorch 0.4.1

***************************************************************

## Reproducibility
We provide the pretrained checkpoints of the SGM model and the SGM+GE model on the RCV1-V2 dataset to help you to reproduce our reported experimental results. The detailed reproduction steps are as follows:

***************************************************************

## Preprocessing
```
python preprocess.py 
```
Remember to download the dataset and put them in the folder *./data/data/*

***************************************************************

## Training
```
python train.py
```

****************************************************************

## Evaluation
```
python predict.py
```

*******************************************************************

## Citation
If you use the above code or the AAPD dataset for your research, please cite the paper:

```
@inproceedings{YangCOLING2018,
  author    = {Pengcheng Yang and
               Xu Sun and
               Wei Li and
               Shuming Ma and
               Wei Wu and
               Houfeng Wang},
  title     = {{SGM:} Sequence Generation Model for Multi-label Classification},
  booktitle = {Proceedings of the 27th International Conference on Computational
               Linguistics, {COLING} 2018, Santa Fe, New Mexico, USA, August 20-26,
               2018},
  pages     = {3915--3926},
  year      = {2018}
}
```
