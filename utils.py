import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import os
import gc

cuda_gpu = torch.cuda.is_available()

def reshape(img, gray_scale_r=False, size_idx = 256):
    if not gray_scale_r:
        return np.reshape(img, (3, size_idx, size_idx))
    else:
        return np.reshape(img, (1, size_idx, size_idx))

# this is stupid 
def pic_normalize(pic):
    pic = np.asarray( pic, dtype=float )
    pic = pic/256

    return pic

def get_video_dir_list(video_path):
    cwd = os.getcwd()
    os.chdir(cwd + video_path[1:])
    dir_list = next(os.walk('.'))[1]
    video_dir_list = []
    for i in dir_list:
        i = video_path + str(i) + '/'
        video_dir_list.append(i)
    os.chdir(cwd)
    video_dir_list.sort()
    return video_dir_list

def get_file_path(video_path):
    frame_paths = []
    for r, d, f in os.walk(video_path):
        for file in f:
            if ".jpg" or ".png" in file:
                filepath = video_path + file
                frame_paths.append(filepath)
    frame_paths.sort()
    return frame_paths

def load_pic(f_num, f_path, normalize = False, gray_scale = False, size_index = 256):
    path_in = f_path[f_num]
    path_target = f_path[f_num+1]

    if gray_scale:
        in_pic = cv.imread(path_in, cv.IMREAD_GRAYSCALE)
        tar_pic = cv.imread(path_target, cv.IMREAD_GRAYSCALE)
    else:
        in_pic = cv.imread(path_in)
        tar_pic = cv.imread(path_target)

    ## normalize pic for 0-256 to 0-1
    if normalize:
        in_pic = pic_normalize(in_pic)
        tar_pic = pic_normalize(tar_pic)
    
    # resize to 256*256 and reshape to tensor
    in_pic = torch.tensor( reshape( cv.resize(in_pic, (size_index, size_index), interpolation=cv.INTER_CUBIC ), gray_scale_r = gray_scale, size_idx = size_index ), dtype=torch.float )
    tar_pic = torch.tensor( reshape( cv.resize(tar_pic, (size_index, size_index), interpolation=cv.INTER_CUBIC ), gray_scale_r = gray_scale, size_idx = size_index ), dtype=torch.float)
    
    if gray_scale == False:
        input_pic = in_pic.view(1, 3, size_index, size_index)
        target_pic = tar_pic.view(1, 3, size_index, size_index)
    else:
        input_pic = in_pic.view(1, 1, size_index, size_index)
        target_pic = tar_pic.view(1, 1, size_index, size_index)
     
    return input_pic, target_pic


def tensor_to_pic(tensor, normalize = False, gray_scale = False, size_index = 256):
    size_idx = size_index
    if cuda_gpu == True:
        tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    if gray_scale == False:
        img = np.reshape(tensor, (size_idx, size_idx, 3))
    else:
        img = np.reshape(tensor, (size_idx, size_idx))

    if normalize == True:     
        img = img*256

    # set to cv format for saving image
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

def check_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        if cuda_gpu == False:
            checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(filename)

        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch

def load_pic_test(f_num, f_path, normalize = False, gray_scale = False, size_index = 256):
    path_in = f_path[f_num]

    if gray_scale:
        in_pic = cv.imread(path_in, cv.IMREAD_GRAYSCALE)
    else:
        in_pic = cv.imread(path_in)

    ## normalize pic for 0-256 to 0-1
    if normalize:
        in_pic = pic_normalize(in_pic)
    
    # resize to 256*256 and reshape to tensor
    in_pic = torch.tensor( reshape( cv.resize(in_pic, (size_index, size_index), interpolation=cv.INTER_CUBIC ), gray_scale_r = gray_scale, size_idx = size_index ), dtype=torch.float )
    
    if gray_scale == False:
        input_pic = in_pic.view(1, 3, size_index, size_index)
    else:
        input_pic = in_pic.view(1, 1, size_index, size_index)
     
    return input_pic

## test
if __name__ == "__main__":
    frame_paths = []
    for r, d, f in os.walk("./origami_single/"):
        for file in f:
            if ".jpg" in file:
                filepath = "./origami_single/" + file
                frame_paths.append(filepath)

    # test gray scale
    test, target = load_pic( 1, frame_paths, normalize=True, gray_scale=True, size_index=128 )
    img = tensor_to_pic(test,normalize=True, gray_scale=True, size_index=128)
    cv.imwrite('color_img.jpg', img)
    cv.imshow('My Image', img)
    cv.waitKey(0)

    # test colorful img
    test, target = load_pic( 1, frame_paths, normalize=True, gray_scale=False )
    img = tensor_to_pic(test,normalize=True, gray_scale=False)
    cv.imwrite('color_img.jpg', img)
    cv.imshow('My Image', img)
    cv.waitKey(0)
