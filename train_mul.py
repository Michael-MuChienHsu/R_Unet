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
#from torchviz import make_dot, make_dot_from_trace
# possible size_index: 2^n, n >= 4, n is int 

# set arguements
args = parse_argument.argrements()
video_path, learn_rate, step, gray_scale_bol = args.videopath, float(args.lr), int(args.step), bool(args.gray_scale)
output_path = args.output_path
epoch_num = int(args.epoch_num)
size_idx = int(args.sz_idx)
loss_function = str(args.loss_func)
predict_frame_num = int(args.predict_frame)
save_img = True

# get lists of frame paths
cwd = os.getcwd()
os.chdir(cwd+video_path[1:])
dir_list = next(os.walk('.'))[1]
video_dir_list = []
for i in dir_list:
    i = video_path + str(i) + '/'
    video_dir_list.append(i)
os.chdir(cwd)

## ste gpu, set data, check gpu, define network, 
gpus = [0]
start_date = str(datetime.datetime.now())[0:10]
cuda_gpu = torch.cuda.is_available()

## if gpu exist, use cuda
if( cuda_gpu ):
    network = torch.nn.DataParallel(net.unet(Gary_Scale = gray_scale_bol, size_index=size_idx), device_ids=gpus).cuda()
else:
    network = net.unet(Gary_Scale = gray_scale_bol, size_index=size_idx)

## get model size
pytorch_total_params = sum(p.numel() for p in network.parameters())
print("number of parameters:", pytorch_total_params)
print("leaening rate:", learn_rate)
print("frame size:", size_idx, 'x', size_idx)
print("input", step, "frames")
print("predict", predict_frame_num, "frames")

## GC memory    
gc.enable()

# set training parameters
optimizer = optim.Adam( network.parameters(), lr = learn_rate )
if loss_function != 'l1':
    critiria = nn.MSELoss()
else:
    critiria = nn.SmoothL1Loss()

loss_list = [] ## records loss through each step in training
batch_size = len(video_dir_list)

for epochs in range(0, epoch_num):
    ## randomly choose tarining video sequence for each epoch
    train_seq = np.random.permutation(batch_size)
    for batch in range(0, batch_size):
        frame_paths = get_file_path(video_dir_list[ train_seq[batch] ])
        new_frame_paths = [ frame_paths[i] for i in range(0, len(frame_paths), 5) ]
        step_size = step + predict_frame_num
        # reset buffer for each video
        buffer = []
        for steps in range(0, step_size):
            #print("epoch", epochs, "steps", steps)
            # Clear the gradients, since PyTorch accumulates them
            start_time = time.time()
            optimizer.zero_grad()

            # load picture, step = pic num
            test, target = load_pic( steps, new_frame_paths, gray_scale=gray_scale_bol, size_index = size_idx)
            if cuda_gpu:
                test = test.cuda()
                target = target.cuda()              

            # Reshape and Forward propagation
            #test = unet_model.reshape(test)
            #pass in buffer with length = steps-1, concatenate latent feature to buffer in network  
            if steps < step:
                output, l_feature = network.forward(test, buffer)
            else:
                print('doing prediction')
                output, l_feature = network.forward(output, buffer)

            #make_dot( output.mean(), params = dict(network.named_parameters() ) )
            #exit()
            # update buffer for storing latent feature
            buffer = buf_update( l_feature, buffer, 6 )

            # Calculate loss
            #loss = critiria( Variable(output.long()),  Variable(target.long()))
            loss = critiria( output, target)

            # record loss in to csv
            loss_value =  float( loss.item() )
            string = 'epoch_' + str(epochs) + '_batch_' + str(batch) + '_step_' + str(steps) 
            loss_list.append( [ string, loss_value ])

            # save img
            if save_img == True :
                if ( (epochs + 1) % 10 == 0) or ( epochs == 0 ):
                    if steps % 1 == 0:
                        output_img = tensor_to_pic(output, gray_scale=gray_scale_bol, size_index = size_idx)
                        output_img_name = './output_2080/' + str(start_date) + '_E' + str(epochs) + '_B'+ str(batch) + '_S'+ str(steps) +'_2output.jpg'
                        cv.imwrite(str(output_img_name), output_img)

            # Backward propagation
            loss.backward(retain_graph = True)

            end_time = time.time()
            elapse_time = round((end_time - start_time), 2)
            
            # Update the gradients
            optimizer.step()
            
            # print memory used
            process = psutil.Process(os.getpid())

            print('epoch', epochs, 'batch', batch, 'step', steps, "loss:", loss, 'time used', elapse_time, 'sec')
            print('used memory', round((int(process.memory_info().rss)/(1024*1024)), 2), 'MB' )

            if cuda_gpu:
                test = test.cpu()
                target = target.cpu()

            del test, target, loss, output, output_img, l_feature
            gc.collect()
            
            #check_tensors()

            if cuda_gpu:
                torch.cuda.empty_cache()
                
        if cuda_gpu:
            torch.cuda.empty_cache()
    # log loss after each epoch
    write_csv_file( './output/'+ start_date +'_loss_record.csv', loss_list )

    # save model
    if (((epochs+1) % 10 ) == 0):
        path = os.getcwd() + '/model1/' + start_date + 'epoch_' + str(epochs) +"_step_" + str(steps) + '_R_Unet.pt'
        torch.save(network.state_dict(), path)
        print('save model to:', path)


    if cuda_gpu:
        torch.cuda.empty_cache()
