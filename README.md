### Intro

Based on: [youtube](https://www.youtube.com/watch?v=GRtgLlwxpc4&ab_channel=DeepLearning)

[youtube](https://www.youtube.com/watch?v=fu2tfOV9vbY&ab_channel=RobMulla)

## Tools Needed
- Git
- Python & pip
- Anaconda (Conda)
- Nvidia CUDA

## Current Setup
- Operating System: Windows 10 Home 64-bit (10.0, Build 19045)
- System Manufacturer: Gigabyte Technology Co., Ltd.
- System Model: B450M DS3H V2
- Processor: AMD Ryzen 7 5700X 8-Core Processor (16 CPUs) ~3.4GHz
- Memory: 16384MB RAM
- Card name: NVIDIA GeForce GTX 1650 SUPER

## Clone THIS repo and go to root
```bash
git clone https://github.com/LuisAlbertoVasquezVargas/medal_detector.git
cd medal_detector
```

## Clone the YOLOv5 repository (in root)
```bash
git clone https://github.com/ultralytics/yolov5
```

## Activate Conda (in /yolov5)
```bash
conda create -n yolov5-env-2 python=3.11
conda activate yolov5-env-2
```

## Install dependencies (in /yolov5)
```bash
pip install -r requirements.txt
```
## Install pytorch (in /yolov5)
[pytorch](https://pytorch.org/get-started/locally/)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Make sure cuda is available (in /yolov5)
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

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

## Label images with makesense.ai [https://makesense.ai/]

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

## Train the model (in /yolov5)

```bash
# nano model 0.25 Hours
python train.py --epochs 800 --data dota.yaml --weights yolov5n.pt --batch-size -1

python detect.py --weights runs/train/exp3/weights/best.pt --conf 0.40 --iou 0.01 --source trainning/match2.jpg

# small model 535 epochs completed in 0.350 hours.
python train.py --epochs 1600 --data dota.yaml --weights yolov5s.pt --batch-size -1

python detect.py --weights runs/train/exp5/weights/best.pt --conf 0.40 --iou 0.01 --source trainning/match2.jpg

# medium model NOT WORKING
python train.py --epochs 1600 --patience 200 --data dota.yaml --weights yolov5s.pt --batch-size -1
```