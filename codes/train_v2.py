import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import R_Unet_ver_4 as net
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
video_path, learn_rate, step, gray_scale_bol, version = args.videopath, float(args.lr), int(args.step), bool(args.gray_scale), args.version
output_path = args.output_path
epoch_num = int(args.epoch_num)
size_idx = int(args.sz_idx)
loss_function = str(args.loss_func)
input_frmae = int( args.input_frame )
skip_frame = int(args.skip_frame)
predict_frame_num = int(args.predict_frame)
laod_model_name = args.load_model_name
Load = args.load

save_img = True

assert ( input_frmae <= step )
assert (os.path.isdir( output_path )) # check output path exist

# get lists of frame paths
all_video_dir_list = get_video_dir_list(video_path)
video_dir_list = all_video_dir_list[0:32]
val_dir_list = all_video_dir_list[32:40]

## ste gpu, set data, check gpu, define network, 
gpus = [0]
start_date = str(datetime.datetime.now())[0:10]
cuda_gpu = torch.cuda.is_available()

## if gpu exist, use cuda
device = torch.device('cuda:0' if cuda_gpu else 'cpu')
network = network_loader(version, gray_scale_bol, size_idx)
network = network.to(device)

## GC memory
gc.enable()

# set training parameters
optimizer = optim.Adam( network.parameters(), lr = learn_rate )

if loss_function != 'l1':
    critiria = nn.MSELoss()
else:
    critiria = nn.SmoothL1Loss()

loss_list = [] ## records loss through each step in training
batch_size = len(video_dir_list) ## batch size = number of avaliable video

# load previous model
if Load == True:
    network, optimizer, start_epoch = load_checkpoint( network, optimizer, laod_model_name )
else:
    start_epoch = 0

# print training info
pytorch_total_params = sum(p.numel() for p in network.parameters())
print("==========================")
print("number of parameters:", pytorch_total_params)
print("leaening rate:", learn_rate)
print("frame size:", size_idx, 'x', size_idx)
print("input", step, "frames")
print("predict", predict_frame_num, "frames")
print("sample every ", skip_frame, "frame(s)")
print("number of epochs", (start_epoch + epoch_num) )
print ("output path", output_path)
print("optimizer", optimizer)
print("val == True")
print("==========================")

input("press enter to continue\n\n")

print("==========================")

for epochs in range(start_epoch, start_epoch + epoch_num):
    ## randomly choose tarining video sequence for each epoch
    train_seq = np.random.permutation(batch_size)[:16] # random train sequence
    train_video_size = len(train_seq)

    if( (epochs) % 50 == 0 ):
        valadation = True
    else:
        valadation = False

    for batch in range(0, train_video_size):
        frame_paths = get_file_path(video_dir_list[ train_seq[batch] ])  
        new_frame_paths = [ frame_paths[i] for i in range(0, len(frame_paths), skip_frame ) ]
        step_size = step + predict_frame_num - 1
        avalible_len = len(new_frame_paths)
        print ('current batch:', video_dir_list[ train_seq[batch] ] )
        # reset buffer for each video
        buffer = []

        start_frame = np.random.randint(0, avalible_len - step_size - 1 )  # random start point

        if avalible_len - start_frame < step_size:
            print( 'not enough image ' )
            pass
        else:
            if( np.random.rand() > 0.5 ):
                exception = True
                step_size = step_size + 1
            else:
                exception = False
                   
            for steps in range( start_frame, start_frame + step_size):
                if (steps == start_frame):
                    free_mem = True
                else:
                    free_mem = False

                #print("epoch", epochs, "steps", steps)
                # Clear the gradients, since PyTorch accumulates them
                start_time = time.time()
                optimizer.zero_grad()

                # load picture, step = pic num
                test, target = load_pic( steps, new_frame_paths, gray_scale = gray_scale_bol, size_index = size_idx)

                
                test = test.to(device)
                target = target.to(device)

                # Reshape and Forward propagation
                #test = unet_model.reshape(test)
                #pass in buffer with length = steps-1, concatenate latent feature to buffer in network
                if steps < start_frame + step:
                    output = network.forward(test, free_mem)
                    if ( steps == start_frame + step - 1 ):
                        print('doing first prediction')
                else:
                    print('doing prediction')
                    output = network.forward(previous_output, free_mem)

                previous_output = output

                #make_dot( output.mean(), params = dict(network.named_parameters() ) )
                #exit()
                # update buffer for storing latent feature
                # buffer = buf_update( l_feature, buffer, step )

                # Calculate loss
                #loss = critiria( Variable(output.long()),  Variable(target.long()))
                loss = critiria( output, target)

                # record loss in to csv
                loss_value =  float( loss.item() )
                string = 'epoch_' + str(epochs) + '_batch_' + str(batch) + '_step_' + str(steps-start_frame)
                loss_list.append( [ string, loss_value ])

                # save img
                if save_img == True or float(loss_value) > 400:
                    if ( (epochs + 1) % 50 == 0) or ( epochs == 0 ) or ( (epochs+1) == ( start_epoch + epoch_num) ):
                        if steps % 1 == 0:
                            output_img = tensor_to_pic(output, normalize=False, gray_scale=gray_scale_bol, size_index = size_idx)
                            output_img_name = output_path + str(start_date) + '_E' + str(epochs) + '_B'+ str(batch).zfill(2) + '_S'+ str(steps-start_frame).zfill(2) +'_output.jpg'
                            cv.imwrite(str(output_img_name), output_img)

                # Backward propagation
                loss.backward(retain_graph = True)

                end_time = time.time()
                elapse_time = round((end_time - start_time), 2)

                # Update the gradients
                optimizer.step()

                # print memory used
                process = psutil.Process(os.getpid())

                print('epoch', epochs, 'batch', batch, 'step', steps-start_frame, "loss:", loss, 'time used', elapse_time, 'sec')
                print('used memory', round((int(process.memory_info().rss)/(1024*1024)), 2), 'MB' )
                print("-------------------------------------")

                gc.collect()

                #check_tensors()

                if cuda_gpu:
                    torch.cuda.empty_cache()

            if exception == True:
               step_size = step_size - 1 
               exception = False

            if cuda_gpu:
                torch.cuda.empty_cache()
    # log loss after each epoch
    write_csv_file( output_path + start_date +'_loss_record.csv', loss_list )


    if valadation == True:
        print("==== validatoin ====\n\n")

        for batch in range(0, len(val_dir_list)):
            frame_paths = get_file_path( val_dir_list[batch] )  
            new_frame_paths = [ frame_paths[i] for i in range(0, len(frame_paths), skip_frame ) ]
            val_start = time.time()
            print(' ----batch{}----  '.format(batch))
            
            for steps in range(0, step_size):
                 test, target = load_pic( steps, new_frame_paths, gray_scale = gray_scale_bol, size_index = size_idx)
                 test = test.to(device)
                 target = target.cuda()

                 if steps < start_frame + step:
                    output = network.forward(test, free_mem)
                    
                 else:
                    output = network.forward(previous_output, free_mem)

                 previous_output = output
                 loss = critiria( output, target)
                 print('val: batch', batch, 'step', steps, "loss:", loss, '\n')
            val_end = time.time()
            print('time used:', round(( val_end - val_start ),2))

    # save model
    if ( ( ( (epochs+1) % 500 ) == 0 ) or ((epochs+1) == ( start_epoch + epoch_num)) or ( (epochs+1)  == 1 ) ):
        path = output_path + start_date + 'epoch_' + str(epochs) +'_R_'+ str(step) + '_P_' + str(predict_frame_num) + '_size_idx_' + str(size_idx) +  '_R_Unet.pt'
        state = { 'epoch': epochs+1, 'state_dict': network.state_dict(), 'optimizer':optimizer.state_dict() }

        torch.save( state, path)
        print('save model to:', path)

    if cuda_gpu:
        torch.cuda.empty_cache()

