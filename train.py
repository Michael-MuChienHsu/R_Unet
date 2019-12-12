import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import R_Unet as net
import numpy as np
import parse_argument
from utils import *
import os
import csv
import datetime
import time
import psutil
import gc

# set arguements
args = parse_argument.argrements()
video_path = args.videopath
learn_rate = float(args.lr)
step = int(args.step)
gray_scale_bol = bool(args.gray_scale)

save_img = True

# get lists of frame paths
frame_paths = get_file_path(video_path)
frame_numbers = len(frame_paths)

## ste gpu, set data, check gpu, define network, 
gpus = [0]
start_date = str(datetime.datetime.now())[0:10]
cuda_gpu = torch.cuda.is_available()
network = net.unet(Gary_Scale = gray_scale_bol)

## if gpu exist, use cuda
if( cuda_gpu ):
    network = torch.nn.DataParallel(network, device_ids=gpus).cuda()

## get model size
pytorch_total_params = sum(p.numel() for p in network.parameters())
print("number of parameters:", pytorch_total_params)

## GC memory 
gc.enable()

# set training parameters
optimizer = optim.Adam( network.parameters(), lr = learn_rate )
critiria = nn.SmoothL1Loss()
#critiria = nn.MSELoss()

loss_list = [] ## records loss through each step in training

for epochs in range(0, 200):
    buffer = []
    for steps in range(0, 10):
        #print("epoch", epochs, "steps", steps)
        # Clear the gradients, since PyTorch accumulates them
        start_time = time.time()
        optimizer.zero_grad()

        # load picture, step = pic num
        test, target = load_pic( steps, frame_paths, gray_scale=gray_scale_bol )
        if cuda_gpu:
            test = test.cuda()
            target = target.cuda()

        # Reshape and Forward propagation
        #test = unet_model.reshape(test)
        #pass in buffer with length = steps-1, concatenate latent feature to buffer in network  
        output, l_feature = network.forward(test, buffer)

        # update buffer for storing latent feature
        buffer = buf_update( l_feature, buffer, 6 )

        # Calculate loss
        #loss = critiria( Variable(output.long()),  Variable(target.long()))
        loss = critiria( output, target)

        # record loss in to csv
        loss_value =  float( loss.item() )
        string = 'epoch_' + str(epochs) + '_step_' + str(steps) 
        loss_list.append( [ string, loss_value ])

        # save img
        if save_img == True :
            output_img = tensor_to_pic(output, gray_scale=gray_scale_bol)
            output_img_name = './output_img2/' + str(start_date) + '_' + str(epochs) + '_' + str(steps) +'_2output.jpg' 
            cv.imwrite(str(output_img_name), output_img)
            del output_img
            del output_img_name

        # Backward propagation
        loss.backward(retain_graph = True)

        end_time = time.time()
        elapse_time = round((end_time - start_time), 2)
        
        print('epoch', epochs, 'step', steps, "loss:", loss, 'time_used', elapse_time)
        # Update the gradients
        optimizer.step()
        
        # print memory used
        process = psutil.Process(os.getpid())
        print('used memory', round((int(process.memory_info().rss)/(1024*1024)), 2), 'MB' )

        if cuda_gpu:
            test = test.cpu()
            target = target.cpu()

        del test
        del target
        gc.collect()

        if (((steps+1) % 10 ) == 0):
            # save model
            path = os.getcwd() + '/model1/' + start_date + 'epoch_' + str(epochs) +"_step_" + str(steps) + '_R_Unet.pt'
            torch.save(network.state_dict(), path)
            print('save model to:', path)

    # log loss after each epoch
    write_csv_file( './output/'+ start_date +'_loss_record.csv', loss_list )

    if cuda_gpu:
        torch.cuda.empty_cache()
