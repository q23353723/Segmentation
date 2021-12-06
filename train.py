import os
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from utils.dataset import Dataset
from utils.preprocessing import *
from model.Unet import UNet
from model.Unet_PlusPlus import Unet_PlusPlus
from tqdm import tqdm
from time import sleep


Fold = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']
DATA_DIR = './datasets'
device = 'cuda'

log = ''

for fold in Fold:
    print('')
    print(fold)
    print('------------------')

    FOLD_DIR = os.path.join(DATA_DIR, fold)
    x_train_dir = os.path.join(FOLD_DIR, 'train')
    y_train_dir = os.path.join(FOLD_DIR, 'trainannot')

    x_valid_dir = os.path.join(FOLD_DIR, 'val')
    y_valid_dir = os.path.join(FOLD_DIR, 'valannot')

    x_test_dir = os.path.join(FOLD_DIR, 'test')
    y_test_dir = os.path.join(FOLD_DIR, 'testannot')

    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing,
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        #augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing,
    )

    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        #augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing,
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    #model = UNet(1, 1).cuda()
    model = Unet_PlusPlus(1, 1).cuda()

    loss = smp.utils.losses.DiceLoss()
    metric = smp.utils.metrics.IoU(threshold=0.5)
    
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    max_iou = 0
    for epoch in range(0, 80):
        print("Epoch:", epoch)
        train_loss = 0
        valid_loss = 0
        train_iou = 0
        valid_iou = 0
        i = 0
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for image, ett_mask, ca_mask in tepoch:
                tepoch.set_description("Train:")
                i+=1

                image, y1, y2 = image.to(device), ett_mask.to(device), ca_mask.to(device)
                optimizer.zero_grad()
                pred_y1, pred_y2 = model(image)
                ett_loss = loss(pred_y1, y1)
                ca_loss = loss(pred_y2, y2)
                dice_loss = ett_loss * 0.5 + ca_loss*0.5

                ett_iou = metric(pred_y1, y1)
                ca_iou = metric(pred_y2, y2)

                train_loss += dice_loss
                train_iou += (ett_iou * 0.5 + ca_iou*0.5)

                dice_loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(dice_loss=train_loss.item()/i, iou_score=train_iou.item()/i)
                sleep(0.1)
        
        i = 0
        model.eval()
        with tqdm(valid_loader, unit="batch") as tepoch:
            for image, ett_mask, ca_mask in tepoch:
                with torch.no_grad():
                    tepoch.set_description("Valid:")
                    i+=1

                    image, y1, y2 = image.to(device), ett_mask.to(device), ca_mask.to(device)
                    pred_y1, pred_y2 = model(image)
                    ett_loss = loss(pred_y1, y1)
                    ca_loss = loss(pred_y2, y2)
                    dice_loss = ett_loss * 0.5 + ca_loss*0.5
                    
                    ett_iou = metric(pred_y1, y1)
                    ca_iou = metric(pred_y2, y2)

                    valid_loss += dice_loss
                    valid_iou += (ett_iou * 0.5 + ca_iou*0.5)
                    
                    tepoch.set_postfix(dice_loss=valid_loss.item()/i, iou_score=valid_iou.item()/i)
                    sleep(0.1)
        
        if valid_iou > max_iou:
            max_iou = valid_iou
            torch.save(model, './model/' + fold +'.pth')
            print("Model Saved!")

        if epoch == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')