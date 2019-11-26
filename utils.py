import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv

def load_image(path):
    img = cv.imread(path)
    return img

def reshape_cv( img ):
    new_img = np.zeros((3, 256, 256), dtype=float)
    for i in range(0, 256):
        for j in range(0, 256):
            for k in range(0, 3):
                new_img[k][i][j] = img[i][j][k]
    
    return new_img

def pic_normalize(pic):
    w = pic.shape[0]
    h = pic.shape[1]
    c = pic.shape[2]

    new_pic = np.zeros((w,h,c), dtype=float)
    for i in range(0, w):
        for j in range(0, h):
            for k in range(0, c):    
                pic[i][j][k] = float(pic[i][j][k])
                new_pic[i][j][k] = pic[i][j][k]/256.0
    
    return new_pic

def load_pic(f_num, f_path):
    path_in = f_path[f_num]
    path_target = f_path[f_num+1]
    in_pic = cv.imread(path_in)
    tar_pic = cv.imread(path_target)
    ## nomalize pic for 0-256 to 0-1
    #in_pic = pic_normalize(in_pic)
    #tar_pic = pic_normalize(tar_pic)

    in_pic = cv.resize(in_pic, (256,256), interpolation=cv.INTER_CUBIC )
    tar_pic = cv.resize(tar_pic, (256,256), interpolation=cv.INTER_CUBIC )
    in_pic = reshape_cv( in_pic )
    tar_pic = reshape_cv( tar_pic)    
    in_pic = torch.tensor( in_pic, dtype=torch.float )
    tar_pic = torch.tensor( tar_pic, dtype=torch.float )
    input_pic = in_pic.view(1, 3, 256, 256)
    target_pic = tar_pic.view(1, 3, 256, 256)
     
    return input_pic, target_pic

def tensor_to_pic(tensor, normalize = True):
    tensor = tensor.view(3, 256, 256)
    ## reshape
    new_img = np.zeros((256, 256, 3), dtype=float)
    for i in range(0, 256):
        for j in range(0, 256):
            for k in range(0, 3):
                new_img[i][j][k] = tensor[k][i][j]
    # anti_normalize
    w = new_img.shape[0]
    h = new_img.shape[1]
    c = new_img.shape[2]
    out_img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(0, w):
        for j in range(0, h):
            for k in range(0, c):
                if normalize == True:
                    out_img[i][j][k] = int(new_img[i][j][k]*256)
                else:
                    out_img[i][j][k] = int(new_img[i][j][k])

    return out_img

def write_csv_file( filename, data):    
    with open(filename, 'w', newline='') as csvfile:
        for i in range(0, len(data)):
            writer = csv.writer(csvfile)
            writer.writerow(data[i])

## test
if __name__ == "__main__":
    img = cv.imread('./origami_single/000.jpg')
    ##cv.imshow('My Image', img)
    ##cv.waitKey(0)
    img = cv.resize(img, (256,256), interpolation=cv.INTER_CUBIC )
    ##cv.imshow('My Image', img)
    ##cv.waitKey(0)
    img = pic_normalize(img)
    img = reshape_cv(img)
    tensor = torch.tensor( img, dtype=torch.float )
    img = tensor_to_pic(tensor)
    cv.imwrite('color_img.jpg', img)
    cv.imshow('My Image', img)
    cv.waitKey(0)
    