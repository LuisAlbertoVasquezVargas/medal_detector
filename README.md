### Intro
Based on: [youtube](https://www.youtube.com/watch?v=GRtgLlwxpc4&ab_channel=DeepLearning)

[text](https://www.youtube.com/watch?v=fu2tfOV9vbY&ab_channel=RobMulla)

## Tools Needed
- Git
- Python & pip
- Anaconda (Conda)
- Nvidia CUDA

## Clone THIS repo
´´´
git clone https://github.com/LuisAlbertoVasquezVargas/medal_detector.git
cd medal_detector
´´´

## Clone the YOLOv5 repository
´´´
git clone https://github.com/ultralytics/yolov5
´´´

## Install dependencies
´´´
cd yolov5
pip install -r requirements.txt
´´´

## Activate Conda
´´´
conda create -n yolov5-env python=3.11
conda activate yolov5-env
´´´

## Make sure cuda is available
´´´
python -c "import torch; print(torch.cuda.is_available())"
´´´

## Create datasets/dota folder

```
Create folder structure
parent
├── yolov5
└── datasets
    └── dota  ← put here the images and labels 
        └── images  
        └── labels
```

## Label images with makesense.ai

## Fill up files to images and labels

## Add the file yolov5/data/dota.yaml

```yaml
path: ../datasets/dota # dataset root dir
train: images # train images (relative to 'path')
val: images # val images (relative to 'path')
test: # test images (optional)

# Classes
names:
  0: TBD
  1: Herald
  2: Guardian
  3: Crusader
  4: Archon
  5: Legend
  6: Ancient
  7: Divine
  8: Inmortal

```

## Train the model

```
python train.py --epochs 800 --data dota.yaml --weights yolov5n.pt --batch-size -1

python detect.py --weights runs/train/exp13/weights/best.pt --conf 0.40 --iou 0.01 --source trainning/match1.jpg
```