import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import os
import shutil
import gc

cuda_gpu = torch.cuda.is_available()

'''
reshape a given image to square image

input: cv img, gray scale: boolean, size index
output: numpy array
'''
def reshape(img, gray_scale_r=False, size_idx = 256):
    if not gray_scale_r:
        return np.reshape(img, (3, size_idx, size_idx))
    else:
        return np.reshape(img, (1, size_idx, size_idx))

'''
given a path, get all child directory from the path
return sorted list
'''
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

'''
given a path, get all 'jpg' and 'png' image
return sorted path
'''
def get_file_path(video_path):
    frame_paths = []
    for r, d, f in os.walk(video_path):
        for file in f:
            if ".jpg" or ".png" or ".pt"  in file:
                filepath = video_path + file
                frame_paths.append(filepath)
    frame_paths.sort()
    return frame_paths

'''
load pics in batch

'''
# return step_size of 4 dimentional tensor, (5 dimentions in total)
def frame_batch_loader(f_start_num, f_path, step_size, normalize = False, gray_scale = False, size_index = 256):
    
    for i in range(0, step_size+1):
        if (i == 0):
            tensor = read_single_pic( f_start_num, f_path, normalize, gray_scale, size_index )
        else:
            next_tensor = read_single_pic( f_start_num + i, f_path, normalize, gray_scale, size_index )
            tensor = torch.cat( (tensor, next_tensor), dim = 0 )

    return tensor


# return 5 dimentional torch tensor
def read_single_pic(f_num, f_path, normalize = False, gray_scale = False, size_index = 256):
    if gray_scale:
        pic = cv.imread(f_path[f_num], cv.IMREAD_GRAYSCALE)
    else:
        pic = cv.imread(f_path[f_num]).transpose(2, 0, 1)

    if normalize:
        pic = pic_normalize(pic)
        
    pic = torch.tensor( cv.resize(pic, (size_index, size_index), interpolation=cv.INTER_CUBIC ), dtype=torch.float)

    if gray_scale:
        pic = pic.view(1, 1, 1, size_index, size_index)
    else:
        pic = pic.view(1, 1, 3, size_index, size_index)

    return(pic)

'''
FOR TRAINING PREDICTION AND SEGMENTATION AT THE SAME TIME
'''

'''
get f_num and f_num + 1 pytorch tensors
for loading tensors, not for pic
'''
def batch_loader( start_num, frame_paths, step_size ):
    
    for i in range( 0, step_size+1 ):
        if i == 0:
            tensor = torch.load( frame_paths[start_num] )
        else:
            next_tensor = torch.load( frame_paths[start_num + i] )
            tensor = torch.cat( ( tensor, next_tensor ), dim = 0 )

    return tensor


def data_loader(f_num, f_path, gray_scale = False, size_index = 256):
    
    test_tensor = torch.load(f_path[f_num])
    target_tensor = torch.load(f_path[f_num+1])

    return test_tensor, target_tensor

def tensor_reshape(tensor, gray_scale = False, size_index = 256, imgflag = True):
    size_idx = size_index
    if cuda_gpu == True:
        tensor = tensor.clone().cpu()
    tensor = tensor.detach().numpy()

    if gray_scale == False:
        img = np.reshape(tensor, (size_idx, size_idx, 3))
    else:
        img = np.reshape(tensor, (size_idx, size_idx))

    # set to cv format for saving image
    if imgflag == True:
        img = np.asarray(img, dtype=np.uint8)
    else:
        img = np.asarray(img, dtype=float)

    return img

def merge_image(tensor, size_index = 128, threshold = 0):
    img = tensor_reshape(tensor[0][0], True, size_index)
    mask = tensor[0][1].reshape( size_index, size_index ).clone().detach().cpu()

    mask = np.asarray( mask*255, dtype = np.uint8  )

    mask_bol = (mask > int(threshold*255) )*1

    if threshold == 0:
        mask = mask*mask_bol
    else:
        mask = mask_bol*255

    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)     

    B, G, R = cv.split(img)
    
    R = R + (255-R)*( mask/255 ) 
    R = np.asarray(R, dtype = np.uint8)

    img = cv.merge( [B, G, R] )

    return img


'''
tensoe to image
'''
def tensor_to_image(tensor, size_index = 128 ):
    img = tensor_reshape(tensor[0][0], True, size_index)
    return img

'''
input a mask tensor and visualize mask only on white background

'''
def mask_image(tensor, size_index = 128, threshold = 0):
    mask = tensor.reshape(size_index, size_index, 1).clone().detach().cpu()
    
    zero_mask1 = torch.zeros_like(mask)
    zero_mask2 = torch.zeros_like(mask)
        
    mask = torch.cat( (zero_mask1, mask), dim = 2 )
    mask = torch.cat( (zero_mask2, mask), dim = 2 ).numpy()
    #print(mask.shape)

    img = np.ones( (size_index, size_index, 3), dtype=np.uint8 )*255

    img = img - 255*mask

    return img

'''
put mask on gray background
'''
def mask_image2(tensor, size_index = 128, threshold = 0):
    background_color = 100
    
    mask =  np.asarray( tensor.reshape(size_index, size_index).clone().detach().cpu(), dtype = float )

    mask_bol = (mask > threshold )*1.0
    
    if threshold == 0:
        mask = mask*mask_bol
    else:
        mask = mask_bol
    
    background = np.asarray(np.ones( (size_index, size_index), dtype=np.uint8 )*background_color, dtype = np.uint8)

    R = np.asarray( background + mask*(255 - background_color), dtype = np.uint8 )

    img = cv.merge( [background, background, R] )

    return img

'''
picture tensor to picture
'''
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

'''
get number after '@' sign
'''
def get_epoch_num( string1 ):
    str_len = len(string1)
    num_flag = False
    num = ''

    for i in range( 0, str_len ):
        char = string1[i]
        if( char == '@' ):
            num_flag = True

        if( num_flag == True ):
            if char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '@']:
                num = num + char
                if i == str_len-1:
                    return num 
            else:
                return num

def check_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

def str2bool(v):
    if v.lower() in ('yes', 'true', 'y', '1'):
        return True
    elif( v.lower() in ('no', 'false', 'n', '0') ):
        return False
    else:
        print('wrong')
        exit()


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
        new_state_dict = {}
        for key, val in checkpoint['state_dict'].items():
            key = key.replace('module.', '')
            new_state_dict[key] = val
        #model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch

'''
check if path exist,
if exist delete and make a new one
else make it
'''
def refresh_dir(path):
    if ( os.path.isdir(path) ):
        print("remove", path)
        shutil.rmtree(path)
    print('make', path)
    os.mkdir(path)

def network_loader(version,  gray_scale_bol, size_idx, gpu_num):
    if version == 'v4' or version == 'V4':
        import R_Unet_ver_4 as net

    elif version == 'M' or version == 'M1' or version == 'm' or version == 'm1' :
        import R_Unet_ver_MB as net
        
    elif version == 'M2' or version == 'm2':
        import R_Unet_ver_M2 as net
        
    elif version == 'MS3' or version == 'M3' or version == 'ms3' or version == 'm3':
        import R_Unet_ver_MS3 as net

    elif version == 'v2_5' or version == 'V2_5':
        import R_Unet_ver_2_5 as net
        
    elif version == 'v2_7' or version == 'V2_7':
        import R_Unet_ver_2_7 as net

    elif version == 'v2' or version == 'V2':
        import R_Unet_ver_2 as net

    else:
        print("please specify correct version.")
        exit()

    network = net.unet(Gary_Scale = gray_scale_bol, size_index=size_idx, gpu_num=gpu_num)
    #network = torch.nn.DataParallel(net.unet(Gary_Scale = gray_scale_bol, size_index=size_idx, gpu_num=gpu_num))

    return network

def str2bool(v):
    if v.lower() in ('yes', 'true', 'y', '1'):
        return True
    elif( v.lower() in ('no', 'false', 'n', '0') ):
        return False
    else:
        raise argparse.ArguementTypeError('Wrong Value')
