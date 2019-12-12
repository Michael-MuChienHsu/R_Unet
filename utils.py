import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import os

def load_image(path):
    img = cv.imread(path)
    return img

def reshape_cv(img, gray_scale_r=False):
    if not gray_scale_r:
        return np.reshape(img, (3, 256, 256))
    else:
        return np.reshape(img, (1, 256, 256))

# this is stupid 
def pic_normalize(pic):
    pic = np.asarray( pic, dtype=float )
    pic = pic/256

    return pic

def get_file_path(video_path):
    frame_paths = []
    for r, d, f in os.walk(video_path):
        for file in f:
            if ".jpg" in file:
                filepath = video_path + file
                frame_paths.append(filepath)
    return frame_paths

def load_pic(f_num, f_path, normalize = False, gray_scale = False):
    path_in = f_path[f_num]
    path_target = f_path[f_num+1]

    if gray_scale == True:
        in_pic = cv.imread(path_in, cv.IMREAD_GRAYSCALE)
        tar_pic = cv.imread(path_target, cv.IMREAD_GRAYSCALE)
    else:
        in_pic = cv.imread(path_in)
        tar_pic = cv.imread(path_target)

    ## normalize pic for 0-256 to 0-1
    if normalize == True:
        in_pic = pic_normalize(in_pic)
        tar_pic = pic_normalize(tar_pic)
    
    # resize to 256*256 and reshape to tensor
    in_pic = torch.tensor( reshape_cv( cv.resize(in_pic, (256,256), interpolation=cv.INTER_CUBIC ), gray_scale_r = gray_scale ), dtype=torch.float )
    tar_pic = torch.tensor( reshape_cv( cv.resize(tar_pic, (256,256), interpolation=cv.INTER_CUBIC ), gray_scale_r = gray_scale), dtype=torch.float )
    
    if gray_scale == False:
        input_pic = in_pic.view(1, 3, 256, 256)
        target_pic = tar_pic.view(1, 3, 256, 256)
    else:
        input_pic = in_pic.view(1, 1, 256, 256)
        target_pic = tar_pic.view(1, 1, 256, 256)
     
    return input_pic, target_pic


def tensor_to_pic(tensor, normalize = False, gray_scale = False):
    tensor = tensor.detach().numpy()
    if gray_scale == False:
        img = np.reshape(tensor, (256, 256, 3))
    else:
        img = np.reshape(tensor, (256, 256))

    if normalize==True:     
        img = img*256

    img = np.asarray(img, dtype=np.uint8)

    return img

def write_csv_file( filename, data):    
    with open(filename, 'w', newline='') as csvfile:
        for i in range(0, len(data)):
            writer = csv.writer(csvfile)
            writer.writerow(data[i])
    
def buf_update( latent_feature, buf, step ):
    if len(buf) < step-1:
        buf.append( latent_feature )
        return buf
    else:
        buf = buf[1:]
        buf.append( latent_feature )
        return buf


## test
if __name__ == "__main__":
    frame_paths = []
    for r, d, f in os.walk("./origami_single/"):
        for file in f:
            if ".jpg" in file:
                filepath = "./origami_single/" + file
                frame_paths.append(filepath)

    # test gray scale
    test, target = load_pic( 1, frame_paths, normalize=True, gray_scale=True )
    img = tensor_to_pic(test,normalize=True, gray_scale=True)
    cv.imwrite('color_img.jpg', img)
    cv.imshow('My Image', img)
    cv.waitKey(0)

    # test colorful img
    test, target = load_pic( 1, frame_paths, normalize=True, gray_scale=False )
    img = tensor_to_pic(test,normalize=True, gray_scale=False)
    cv.imwrite('color_img.jpg', img)
    cv.imshow('My Image', img)
    cv.waitKey(0)
