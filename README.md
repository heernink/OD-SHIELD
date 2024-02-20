# OD-SHEILD : Convolutional Autoencoder-based Defense against Adversarial Patch Attacks in Object Detection
PyTorch Implementation of __OD-SHIELD__ 
<p align="center">
  <img src="https://github.com/heernink/OD-SHIELD/assets/75311780/e32fde16-008d-4494-bc5c-cd760a70c0ac" width="600" height="400"/>
</p>


## Datasets
To learn OD-SHIELD, three datasets were used: COCO DATASET, VISDRONE DATASET, and ARGOVERSE DATASET. While the COCO DATASET contains various objects, the ARGOVERSE DATASET and VISDRONE DATASET are focused on traffic scenarios. Therefore, to effectively defend against attacks in autonomous driving situations, OD-SHIELD were trained by reducing the COCO DATASET classes to four: HUMAN, CAR, TRUCK, and BUS (the code for this preprocessing is not provided). When you run the `download_dataset.sh` script, the datasets will be downloaded to the given path, and for training OD-SHIELD, the downloaded datasets should be structured with subfolders under ./datasets directory: ./datasets/images and ./datasets/labels.
```
./download_dataset.sh
```
⚠ Error occur in `download_dataset.sh`. Thus, we recommend to enter below link and download, unzip the zipfile.<br>
[Link](https://drive.usercontent.google.com/download?id=1st9qW3BeIwQsnR0t8mRpvbsSWIo16ACi&export=download)

## Training 
OD-SHIELD provides diversity in experiments through the addition of various arguments via the `train.py` file. The paths to the image files and label files of the dataset are provided as arguments following `--data_path` and `--label_path`, respectively. Additionally, batch size, learning rate, and the number of epochs for the dataset can be specified by arguments following `--bs`, `--lr`, and `--epochs`, allowing experiments to be conducted tailored to each computing environment. The trained model is saved at the path specified by `--checkpoint_path`, with the model's training being saved by default every 100 iterations. If visualizing the training process of the model through TensorBoard is desired, the `--visualize` flag can be set, and the training records can be received in log file format from `--log_path`. 
<p align="center">
  <img src="https://github.com/heernink/OD-SHIELD/assets/75311780/d9f7a957-78d8-41b9-ba94-9947e469c0cf" width="800" height="300"/>
</p>


## Patch
OD-SHIELD utilizes various adversarial patches for training. The paths to these patches are located in the directory `./patch/patch_sample`, and patches are randomly selected from here for training. If the user has custom patches for training, they can specify the paths to these custom patches in `train.py` using the `--patch_paths` argument. Below is an example training code file. 

```
train.py --bs 16 --lr 0.001 --epochs 1200 --gpu_id 0 --checkpoint_path ./checkpoints --log_path ./logs --visualize --data_path ./datasets/images --label_path ./datasets/labels --patch_paths ./patch/patch_sample 
```
---
## Evaluating
To compare the performance of the trained OD-SHIELD models, you can run `defense.py`. When running `defense.py`, you need to add the `--od_shiled` argument to enable object detection with OD-SHIELD applied. Without this argument, you can evaluate the model's performance when subjected to adversarial attacks. By providing this argument, you can assess how effectively the model defends against these attacks. 
```

# 1. Adversarial Patch Attack
python defense.py --device 0 --workers 0 --batch_size 16 --data data/custom.yaml --weights ./best.pt --model_path 

# 2. Adversarial Patch Attack Defense
python defense.py --device 0 --workers 0 --batch_size 16 --data data/custom.yaml --weights ./best.pt --od_shiled

```
|Defense|Dataset|mAP_50|Dataset|mAP_50|Dataset|mAP_50|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|No Defense|COCO|0.487|VISDRONE|0.209|ARGOVERSE|0.362|
|[FQ](http://www.naver.com)|COCO|0.486|VISDRONE|0.210|ARGOVERSE|0.370|
|[JPEG](http://www.naver.com)|COCO|0.426|VISDRONE|0.136|ARGOVERSE|0.264|
|[LGS](http://www.naver.com)|COCO|0.425|VISDRONE|0.163|ARGOVERSE|0.335|
|[SAC](http://www.naver.com)|COCO|0.490|VISDRONE|0.215|ARGOVERSE|0.390|
|**OD-SHIELD**|COCO|**0.563**|VISDRONE|**0.307**|ARGOVERSE|**0.425**|
