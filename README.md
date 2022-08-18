# YOLOL: Towards Robust Mandibular Molar Landmark Detection on CT Volumes

![alt text](images/cover.png "Title")

## 1. Introduction
### 1.1 What for?

Recent 3D landmark detection methods usually aim to detect landmarks of fixed numbers. However, detecting landmarks of fixed numbers does not satisfy the real clinical needs because tissues or organs may be damaged or absent in the CT volumes. This situation requires the model first to discriminate the existence of landmarks, then to locate the recognized landmarks. 

The project aims to identify the anatomy locations of the second and third mandibular molars' crowns and roots. The task has two main challenges:

* Mandibular molars have different root numbers because of molars' variant growth.

    <img src="images/problem1.png" alt="isolated" width="300"/>

* Mandibular molars can be damaged by dental diseases, trauma, or surgery.

    <img src="images/problem2.png" alt="isolated" width="500"/>

  
### 1.2 HighLights
* A novel landmark regression loss is proposed by predicting offsets to anchor balls.
* For landmark classification, the online hard negative mining is used for counting loss of nonexistent landmarks and a small regularization constraint loss is performed for voxels outside the anchor balls.
* In the inference stage, a landmark voting method by caculating the minimum distances among landmark candidates is used for selecting final accurate landmarks.
* The proposed method presents good performance on 648 CT volumes, and the dataset is public available.


## 2. Preparation
### 2.1 Requirements
- python >=3.7
- pytorch >=1.10.0
- Cuda 10 or higher
- numpy
- pandas
- scipy
- nrrd
- time

### 2.2 Data Preparation
The dataset is available at https://drive.google.com/file/d/1NGsBbqXZLDlkiSJtDQdyMlXzgnkFoVON/view?usp=sharing
* Data division
```
    - mmld_dataset/train     # 458 samples for training
    - mmld_dataset/val       # 100 samples for validation
    - mmld_dataset/test      # 100 samples for testing
```
* Data format
```
    - *_volume.nrrd     # 3D volumes
    - *_label.npy       # landmarks
    - *_spacing.npy     # CT spacings, used for calculating MRE
```

## 3. Train and Test
### 3.1 Network Training 

* Training with different network backbone
```
python main_yolol.py --model_name PVNet               # network training using backbone PVNet
python main_yolol.py --model_name PUNet3D             # network training using backbone PUNet3D
python main_yolol.py --model_name PResidualUNet3D     # network training using backbone PResidualUNet3D
``` 

* Training with different GPUs
```
python main_yolol.py --gpu 0         # training with 1 gpu
python main_yolol.py --gpu 0,1,2,3   # training with 4 gpus
```

### 3.2 Fine-tuning in pretrained checkpoint
```
python main_yolol.py --resume ../SavePath/yolol/model.ckpt
```

### 3.3 Metric counting
```
python main_yolol.py --test_flag 0 --resume ../SavePath/yolol/model.ckpt  # calculate MRE and SDR in validation set
python main_yolol.py --test_flag 1 --resume ../SavePath/yolol/model.ckpt  # calculate MRE and SDR in test set
```

### 3.4 Training baseline heatmap regression model
```
python main_baseline.py   # network training for baseline heatmap regression model 
```

## 4. Contact


Institution: Intelligent Medical Center, Sichuan University

email: tao_he@scu.edu.cn; taohescu@gmail.com

## 5. Citation (coming soon)

## 6. FAQ (coming soom)

