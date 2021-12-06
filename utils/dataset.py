import numpy as np
import os
import cv2
import imgaug.augmenters as iaa
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.ett_masks_fps = [os.path.join(os.path.join(masks_dir, 'ett'), image_id) for image_id in self.ids]
        self.ca_masks_fps = [os.path.join(os.path.join(masks_dir, 'carina'), image_id) for image_id in self.ids]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i], 0)
        ett_mask = cv2.imread(self.ett_masks_fps[i], 0)
        ca_mask = cv2.imread(self.ca_masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(ett_mask > 0)]
        ett_mask = np.stack(masks, axis=-1).astype('float')
        masks = [(ca_mask > 0)]
        ca_mask = np.stack(masks, axis=-1).astype('float')

        seq = iaa.Sequential([
              iaa.MaxPooling(3,keep_size=False),
              iaa.CropToFixedSize(width=384, height=512, position=(0.5,0.85)),
          ])
        image = seq(images=[image])[0]
        ett_mask = seq(images=[ett_mask])[0]
        ca_mask = seq(images=[ca_mask])[0]
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, masks=[ett_mask, ca_mask])
            image, ett_mask, ca_mask = sample['image'], sample['masks'][0], sample['masks'][1]
        
        # apply preprocessing
        if self.preprocessing:
            image, ett_mask, ca_mask = self.preprocessing(image=image, ett_mask=ett_mask, ca_mask=ca_mask)
        
        return image, ett_mask, ca_mask
        
    def __len__(self):
        return len(self.ids)