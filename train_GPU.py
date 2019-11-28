import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import R_Unet_GPU as net
import numpy as np
import parse_argument
from utils import *
import os
import csv
import datetime
import time

import gc
import pprint
from pympler.tracker import SummaryTracker
tracker = SummaryTracker()

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
network = network.cuda()

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
loss_list

for epochs in range(0, 200):
    inpt = [0,0]
    network.lstm_buffer = []
    for steps in range(0, 10):
        #print("epoch", epochs, "steps", steps)
        # Clear the gradients, since PyTorch accumulates them
        start_time = time.time()
        optimizer.zero_grad()

        print("1")
        tracker.print_diff()

        # load picture, step = pic num
        inpt[0], inpt[1] = load_pic( steps, frame_paths )

        if cuda_gpu:
            inpt[0] = inpt[0].cuda()
            inpt[1] = inpt[1].cuda()

        print("2")
        tracker.print_diff()

        # Reshape and Forward propagation
        #test = unet_model.reshape(test)
        output = network.forward(inpt[0])

        print("3")
        tracker.print_diff()

        # Calculate loss
        #loss = critiria( Variable(output.long()),  Variable(target.long()))
        loss = critiria( output, inpt[1])


        print("4")
        tracker.print_diff()

        # record loss in to csv
        loss_value =  float( loss.item() )
        string = 'epoch_' + str(epochs) + '_step_' + str(steps)
        #loss_list.append( [ string, loss_value ])
        #write_csv_file( './output/'+ start_date +'_loss_record.csv', loss_list )

        print("5")
        tracker.print_diff()

        # Backward propagation
        loss.backward(retain_graph=True)

        print("6")
        tracker.print_diff()

        end_time = time.time()
        elapse_time = round((end_time - start_time), 2)

        print("7")
        tracker.print_diff()

        print('epoch', epochs, 'step', steps, "loss:", loss, 'time_used', elapse_time)
        # Update the gradients
        optimizer.step()

        print("8")
        tracker.print_diff()

        if ((steps+1) % 1000000000 == 0):
            # transfer output from tensor to image
            out_img = tensor_to_pic( output )  
            # save image
            save_string = './Output_img/' + str(epochs) + '_' + str( frame_paths[steps][len(frame_paths[steps])-7:] ) 
            cv.imwrite(save_string , out_img)
            # save model
            path = os.getcwd() + '/model1/epoch_' + str(epochs) +"_step_" + str(steps) + '_R_Unet.pt'
            torch.save(network.state_dict(), path)
            print('save model to:', path)

        if cuda_gpu:
            inpt[0] = inpt[0].cpu()
            inpt[1] = inpt[1].cpu()
            inpt[0] = 0
            inpt[1] = 0

        n = gc.collect()
        print("Unreachable objects: ", n)
        print("Remaining Garbage: ")
        pprint.pprint(gc.garbage)

        print("9")
        tracker.print_diff()
