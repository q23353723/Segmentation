import albumentations as albu
import numpy as np
import torch

def get_training_augmentation():
    train_transform = [
        #albu.PadIfNeeded(384, 384),
        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.15, rotate_limit=5, shift_limit=0.1, p=1, border_mode=0),

        albu.CLAHE(clip_limit=[1,4],p=0.7),
    ]
    return albu.Compose(train_transform)

def normalization1(x, **kwargs):
    return (x - np.mean(x)) / np.std(x)

def to_tensor(x, **kwargs):
    x = torch.from_numpy(x).float()
    return x

def to_tensor2(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')  #RGB

def add_dimention(x, **kwargs):
    return np.expand_dims(x, axis=0)

def get_preprocessing(image, ett_mask, ca_mask):
    image = normalization1(image)
    image = add_dimention(image)
    image, ett_mask, ca_mask = to_tensor(image), to_tensor2(ett_mask), to_tensor2(ca_mask)
    return image, ett_mask, ca_mask