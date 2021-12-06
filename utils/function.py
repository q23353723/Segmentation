import numpy as np
import cv2
import torch

#Find ETT endpoint
def find_point(image):
    image = np.where(image > 0.5, 255, 0)
    image = np.array(image,np.uint8)
    ret,binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = [cv2.contourArea(a) for a in contours]
    if area == []:
        return torch.tensor(0).float()
    contour = contours[area.index(max(area))].squeeze()
    contour = contour[contour[:, 1].argsort()]
    point = contour[-1]
    return torch.tensor(point[1]).float()

#Find carina tip point
def get_carina_point(img):
    img = np.where(img > 0.5, 255, 0)
    img = np.uint8(img)
    contours, hierarchies = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0,255,0), 0)
    area = [cv2.contourArea(a) for a in contours]
    if area == []:
        return torch.tensor(0).float()
    cnt = contours[area.index(max(area))]
    cnt = cnt.squeeze()
    cnt = sorted(cnt, key = lambda c : c[1])
    length = int((cnt[-1][1] - cnt[0][1]) * 0.3 + cnt[0][1])
    for i in range(len(cnt)):
        for j in range(len(cnt)):
            if(cnt[i][1] == cnt[j][1] and cnt[i][0] != cnt[j][0] and cnt[i][1] > length):
                r = sorted([cnt[i][0],cnt[j][0]])
                if img[cnt[i][1],r[0]:r[1]].any() == False and len(img[cnt[i][1],r[0]:r[1]]) > 3: #and img[cnt[i][1],r[0]-1:r[0]].all() and img[cnt[i][1],r[1]:r[1]+1].all()
                    #x = (cnt[i][0] + cnt[j][0]) // 2
                    y = cnt[i][1]
                    return torch.tensor(y).float()
