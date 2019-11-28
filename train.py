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

# set arguements
args = parse_argument.argrements()
video_path = args.videopath
step = int(args.step)

# enumerate photos (frames) in a diractory and save names in list: files
frame_paths = []
for r, d, f in os.walk(video_path):
    for file in f:
        if ".jpg" in file:
            filepath = video_path + file
            frame_paths.append(filepath)

frame_numbers = len(frame_paths)
'''
pic = cv.imread(frame_paths[10])
print(pic.shape)
r_pic = cv.resize(pic, (256,256), interpolation=cv.INTER_CUBIC )
print(r_pic.shape)
'''
'''
test, target = load_pic( 4, frame_paths )
cv.imshow( 'target', tensor_to_pic(target) )
cv.waitKey()
exit()
'''
## ste gpu, set mete info, check gpu, define network, 
gpus = [0]
start_date = str(datetime.datetime.now())[0:10]
cuda_gpu = torch.cuda.is_available()
network = net.unet()

## check gpu status
if( cuda_gpu ):
    network = torch.nn.DataParallel(network, device_ids=gpus).cuda()

## get model size
pytorch_total_params = sum(p.numel() for p in network.parameters())
print("number of parameters:", pytorch_total_params)

# set parameters
optimizer = optim.Adam( network.parameters(), lr = 0.0001 )
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
        test, target = load_pic( steps, frame_paths )
        if cuda_gpu:
            test = test.cuda()
            target = target.cuda()

        # Reshape and Forward propagation
        #test = unet_model.reshape(test)
        output, buffer = network.forward(test, buffer)

        # Calculate loss
        #loss = critiria( Variable(output.long()),  Variable(target.long()))
        loss = critiria( output, target)

        # record loss in to csv
        loss_value =  float( loss.item() )
        string = 'epoch_' + str(epochs) + '_step_' + str(steps) 
        loss_list.append( [ string, loss_value ])
        write_csv_file( './output/'+ start_date +'_loss_record.csv', loss_list )

        # Backward propagation
        loss.backward(retain_graph=True)

        end_time = time.time()
        elapse_time = round((end_time - start_time), 2)
        
        print('epoch', epochs, 'step', steps, "loss:", loss, 'time_used', elapse_time)
        # Update the gradients
        optimizer.step()

        if cuda_gpu:
            test = test.cpu()
            target = target.cpu()

        if ((steps+1) % 10 == 0):
            # transfer output from tensor to image
            out_img = tensor_to_pic( output )  
            # save image
            save_string = './Output_img/' + str(epochs) + '_' + str( frame_paths[steps][len(frame_paths[steps])-7:] ) 
            cv.imwrite(save_string , out_img)
            # save model
            path = os.getcwd() + '/model1/epoch_' + str(epochs) +"_step_" + str(steps) + '_R_Unet.pt'
            torch.save(network.state_dict(), path)
            print('save model to:', path)
