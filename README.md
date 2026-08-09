# TF2 Class Classifier Dataset

An image classification dataset for identifying Team Fortress 2 character
classes (9 classes), with exactly one character per image.

## Source
This dataset is derived from the [TF2 object detection dataset on Roboflow](https://universe.roboflow.com/kerem-ozturk/team-fortress-2-classes-nuo4m-oeuya)
by cropping each bounding box into its own image (see `crop_dataset.py`).
Images are screenshots from Team Fortress 2 (Valve Corporation game content).
This project is for educational/research purposes only, not for commercial
use, and is not affiliated with or endorsed by Valve Corporation.

## Dataset Stats
- Total images: 929
- Classes: 9
- Split: 643 train / 184 valid / 102 test

## Structure
cropped_dataset/{train,valid,test}/{class_name}/*.jpg

Classes: demoman, engineer, heavy, medic, pyro, scout, sniper, soldier, spy

## Usage
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train_ds = datasets.ImageFolder("cropped_dataset/train", transform=transform)
val_ds   = datasets.ImageFolder("cropped_dataset/valid", transform=transform)

## How it was generated
Original object detection annotations (YOLO format) were used to crop each
labeled character out of its source screenshot, pad it slightly, and resize
it with letterboxing to 224x224.
