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
squeeze image to [0, 1]
'''

# this is stupid 
def pic_normalize(pic):
    pic = np.asarray( pic, dtype=float )
    pic = pic/256

    return pic

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
f_num: index number for image
f_path: list of all file(picture) path

get number (f_num) and (f_num + 1) jps or png images as pair
return test_pic, target_pic

test_pic: : pytorch tensor, f_path[ f_num ]
target_pic: : pytorch tensor, f_path[ f_num + 1 ]
'''
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

'''
better load pic method,
old one simply reshape, this one moves channel from height, width, channel to channel, height, width
for gray scale, its the same, but different for color image.
'''
def load_pic2(f_num, f_path, normalize = False, gray_scale = False, size_index = 256):
    path_in = f_path[f_num]
    path_target = f_path[f_num+1]

    if gray_scale:
        in_pic = cv.imread(path_in, cv.IMREAD_GRAYSCALE).transpose(2, 0, 1)
        tar_pic = cv.imread(path_target, cv.IMREAD_GRAYSCALE).transpose(2, 0, 1)
    else:
        in_pic = cv.imread(path_in).transpose(2, 0, 1)
        tar_pic = cv.imread(path_target).transpose(2, 0, 1)


    ## normalize pic for 0-256 to 0-1
    if normalize:
        in_pic = pic_normalize(in_pic)
        tar_pic = pic_normalize(tar_pic)
    
    # resize to 256*256 and reshape to tensor
    in_pic = torch.tensor(  cv.resize(in_pic, (size_index, size_index), interpolation=cv.INTER_CUBIC ),  dtype=torch.float )
    tar_pic = torch.tensor( cv.resize(tar_pic, (size_index, size_index), interpolation=cv.INTER_CUBIC ), dtype=torch.float)
    
    if gray_scale == False:
        input_pic = in_pic.view(1, 3, size_index, size_index)
        target_pic = tar_pic.view(1, 3, size_index, size_index)
    else:
        input_pic = in_pic.view(1, 1, size_index, size_index)
        target_pic = tar_pic.view(1, 1, size_index, size_index)
     
    return input_pic, target_pic


'''
get f_num and f_num + 1 pytorch tensors

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
'''
def merge_image(tensor, size_index = 128):
    img =tensor_reshape(tensor[0][0], True, size_index)
    mask = tensor_reshape(tensor[0][1], True, size_index, False)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)


    for i in range(0, size_index):
        for j in range(0, size_index):
            if mask[i][j] > 0.1:
                img[i][j][2] = ( img[i][j][2] + int(255*mask[i][j]) ) / ( 1 + mask[i][j] ) 
            
    return img
'''
'''
26 times faster than previous one
'''

def merge_image(tensor, size_index = 128):
   
    img = tensor_reshape(tensor[0][0], True, size_index)
    mask = tensor[0][1].reshape( size_index, size_index ).clone().detach().cpu()
    mask = np.asarray( mask*255, dtype = np.uint8  )
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)     

    B, G, R = cv.split(img)
    
    R = R + (255-R)*( mask/255 ) 
    R = np.asarray(R, dtype = np.uint8)

    img = cv.merge( [B, G, R] )

    return img


'''
input a mask tensor and visualize mask only on white background

'''
def mask_image(tensor, size_index = 128):
    mask = tensor.reshape(size_index, size_index, 1).clone().detach().cpu()
    zero_mask1 = torch.zeros_like(mask)
    zero_mask2 = torch.zeros_like(mask)
        
    mask = torch.cat( (zero_mask1, mask), dim = 2 )
    mask = torch.cat( (zero_mask2, mask), dim = 2 ).numpy()
    #print(mask.shape)

    img = np.ones( (size_index, size_index, 3), dtype=np.uint8 )*255

    img = img - 255*mask

    return img

def tensor_to_image(tensor, size_index = 128):
    img =tensor_reshape(tensor[0][0], True, size_index)
    return img

'''
put mask on gray background
'''
def mask_image2(tensor, size_index = 128):
    background_color = 100
    
    mask = tensor.reshape(size_index, size_index, 1).clone().detach().cpu()
    zero_mask1 = torch.zeros_like(mask)
    zero_mask2 = torch.zeros_like(mask)
    
    mask = torch.cat( (zero_mask1, mask), dim = 2 )
    mask = torch.cat( (zero_mask1, mask), dim = 2 ).numpy()

    background = np.ones( (size_index, size_index, 3), dtype=np.uint8 )*background_color 

    img = background + mask*(255 - background_color)

    img = np.asarray( img, dtype = np.uint8 )

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
        model.load_state_dict(checkpoint['state_dict'])
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

'''
testing code
now used yet

'''
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

def network_loader(version,  gray_scale_bol, size_idx):
    if version == 'v4' or version == 'V4':
        import R_Unet_ver_4 as net

    elif version == 'M' or version == 'M1' or version == 'm' or version == 'm1' :
        import R_Unet_ver_M as net
        
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

    network = torch.nn.DataParallel(net.unet(Gary_Scale = gray_scale_bol, size_index=size_idx))

    return network

def str2bool(v):
    if v.lower() in ('yes', 'true', 'y', '1'):
        return True
    elif( v.lower() in ('no', 'false', 'n', '0') ):
        return False
    else:
        raise argparse.ArguementTypeError('Wrong Value')

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
