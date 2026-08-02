# Fresh vs Rotten Fruit Classification (CNN)

A simple TensorFlow/Keras CNN for binary image classification.

## Features
- Binary classification (Fresh vs Rotten)
- Image preprocessing & augmentation
- CNN built with TensorFlow/Keras
- Saves trained model as `fresh_rotten_classifier.h5`

## Dataset Structure
```
dataset/
├── train/
│   ├── Fresh/
│   └── Rotten/
└── test/
    ├── Fresh/
    └── Rotten/
```

Place your images in the folders above.

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python train.py
```
