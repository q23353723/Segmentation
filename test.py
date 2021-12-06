import os
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from utils.dataset import Dataset
from utils.preprocessing import *
from utils.loss import *
from tqdm import tqdm
from time import sleep

Fold = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']
DATA_DIR = './datasets'
device = 'cuda'

for fold in Fold:
    print('')
    print(fold)
    print('------------------')

    FOLD_DIR = os.path.join(DATA_DIR, fold)

    x_test_dir = os.path.join(FOLD_DIR, 'test')
    y_test_dir = os.path.join(FOLD_DIR, 'testannot')

    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        #augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing,
    )

    test_dataloader = DataLoader(test_dataset)

    model = torch.load('./model/' + fold + '.pth')

    loss = smp.utils.losses.DiceLoss()
    ETT_CM = CM()
    ETT_YN = YN() #0.5
    ETT_YN2 = YN2() #1
    CA_CM = CM2()
    CA_YN = YN3() #0.5
    CA_YN2 = YN4() #1
    metric = smp.utils.metrics.IoU(threshold=0.5)
    
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    total_loss = 0
    total_iou = 0
    total_ett_cm = 0
    total_ett_YN = 0 #0.5
    total_ett_YN2 = 0 #1
    total_ca_cm = 0
    total_ca_YN = 0 #0.5
    total_ca_YN2 = 0 #1
        
    i = 0
    model.eval()
    with tqdm(test_dataloader, unit="batch") as tepoch:
        for image, ett_mask, ca_mask in tepoch:
            with torch.no_grad():
                tepoch.set_description("Test")
                i+=1

                image, y1, y2 = image.to(device), ett_mask.to(device), ca_mask.to(device)
                pred_y1, pred_y2 = model(image)
                ett_mask = np.where(pred_y1.cpu().numpy() > 0.5, 255, 0).squeeze()
                ca_mask = np.where(pred_y2.cpu().numpy() > 0.5, 255, 0).squeeze()
                #print(ett_mask.shape)
                #cv2.imwrite('C:/Users/IDSL/Desktop/Lab/Youda/result/two_seg/' + 'ett_' + test_dataset.ids[i-1], ett_mask)
                #cv2.imwrite('C:/Users/IDSL/Desktop/Lab/Youda/result/two_seg/' + 'ca_' + test_dataset.ids[i-1], ca_mask)

                ett_loss = loss(pred_y1, y1)
                ca_loss = loss(pred_y2, y2)
                dice_loss = ett_loss * 0.5 + ca_loss*0.5
                
                ett_iou = metric(pred_y1, y1)
                ca_iou = metric(pred_y2, y2)

                total_loss += dice_loss.item()
                total_iou += (ett_iou * 0.5 + ca_iou*0.5).item()
                total_ett_cm += ETT_CM(pred_y1, y1).item()
                total_ett_YN += ETT_YN(pred_y1, y1).item()
                total_ett_YN2 += ETT_YN2(pred_y1, y1).item()
                total_ca_cm += CA_CM(pred_y2, y2).item()
                total_ca_YN += CA_YN(pred_y2, y2).item()
                total_ca_YN2 += CA_YN2(pred_y2, y2).item()
                
                tepoch.set_postfix(dice_loss=total_loss/i, iou_score=total_iou/i, ETT_CM=total_ett_cm/i, ETT_YN=total_ett_YN/i, ETT_YN2=total_ett_YN2/i, CA_CM=total_ca_cm/i, CA_YN=total_ca_YN/i, CA_YN2=total_ca_YN2/i)
                sleep(0.1)