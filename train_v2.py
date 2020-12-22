import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import parse_argument
from utils import *
from tensorboardX import SummaryWriter
import os
import csv
import datetime
import time
import psutil
import gc
#from torchviz import make_dot, make_dot_from_trace
# possible size_index: 2^n, n >= 4, n is int 

# set arguements
'''
video_path:        directory of video frames
learn_rate:        learning rate
gray_scale_bol:    boolean, True for use gray scale image, False for use color image
version:           runet version to use
output_path:       directory for output image, model and tensorboard summary
epoch_num:         number of epochs to train
size_idx:          resize image to size_idx x size_idx square image
loss_function:     loss funtion. L1 loss as default
skip_frame:        sample rate, to reduce compactness
predict_frame_num: number of prections    
Load:              True: load model from checkpoint; False: start from zero
load_model_name:   checkpoint name to load and train 
'''

args = parse_argument.argrements()
video_path, learn_rate, step, gray_scale_bol, version = args.videopath, float(args.lr), int(args.step), bool(args.gray_scale), args.version
output_path = args.output_path
epoch_num = int(args.epoch_num)
size_idx = int(args.sz_idx)
loss_function = str(args.loss_func)
#input_frame = int( args.input_frame )
skip_frame = int(args.skip_frame)
predict_frame_num = int(args.predict_frame)

assert (os.path.isdir( output_path )) # check output path exist

# if gpu exist, use cuda
gpu_num = args.gpu
device = torch.device('cuda:'+str(gpu_num) if cuda_gpu else 'cpu')

# load network
network = network_loader(version, gray_scale_bol, size_idx, gpu_num)
network = network.to(device)

# set optimizer and Loss fuction
optimizer = optim.Adam( network.parameters(), lr = learn_rate )
writer = SummaryWriter(output_path + 'Summary_writer')

# load model from check point or start from epoch 0
Load = args.load
if Load == True:
    load_model_name = args.load_model_name
    if os.path.isfile( load_model_name ):
        network, optimizer, start_epoch = load_checkpoint( network, optimizer, load_model_name )
    else:
        print("checkpoint do not exist, set Load to False or specify correct checkpoint path")
        exit()
else:
    start_epoch = 0


# save image while training or not
save_img = True

# get lists of frame paths
all_video_dir_list = get_video_dir_list(video_path)

video_dir_list = all_video_dir_list[0:160]
val_dir_list = all_video_dir_list[160:180]

# ste gpu, set data, check gpu, define network, 
gpus = [0]
start_date = str(datetime.datetime.now())[0:10]
cuda_gpu = torch.cuda.is_available()

# Garbage Collector
gc.enable()

if loss_function == 'MSE' or loss_function == 'mse':
    critiria = nn.MSELoss()
else:
    critiria = nn.SmoothL1Loss()

loss_list = [] ## records loss through each step in training
train_video_num = len(video_dir_list) # batch size = number of avaliable video, and later 

# print training info
pytorch_total_params = sum(p.numel() for p in network.parameters())
print("==========================")
print("model version:", version)
print("training/validation video path", video_path)
print("number of parameters:", pytorch_total_params)
print("leaening rate:", learn_rate)
print("frame size:", size_idx, 'x', size_idx)
print("input", step, "frames")
print("predict", predict_frame_num, "frames")
print("sample every ", skip_frame, "frame(s)")
print("number of total epochs", (start_epoch + epoch_num) )
print("output path", output_path)
print("optimizer", optimizer)
print("val == True")
print("==========================")

input("press enter to continue\n\n")

print("==========================")

for epochs in range(start_epoch, start_epoch + epoch_num):
    # randomly choose 16 videos from video pool as training video for this epoch 
    train_seq = np.random.permutation(train_video_num)[:16] # random train sequence
    train_video_size = len(train_seq)
    print('epoch_num', epochs, '/', start_epoch + epoch_num)

    # run validation every 50 epoches 
    if( (epochs) % 50 == 0 ):
        validation = True
    else:
        validation = False

    for batch in range(0, train_video_size):
        frame_paths = get_file_path(video_dir_list[ train_seq[batch] ])  
        new_frame_paths = [ frame_paths[i] for i in range(0, len(frame_paths), skip_frame ) ]
        # for number 10th predictions, we only need to do 9 times
        # eg. input t = 0, output t = 1
        #     input t = 1, output t = 2
        #                 .
        #                 .
        #                 .
        #     input t = 8, output t = 10
        # so step_size = step + predict_frame_num - 1
        # step_size:         step per batch
        # step:              input ground truth frame number
        # predict_frame_num: frame munber for prediction
        step_size = step + predict_frame_num - 1
        #print ('current batch:', video_dir_list[ train_seq[batch] ] )

        avalible_len = len(new_frame_paths)
        start_frame = np.random.randint(0, avalible_len - step_size - 2 )  # random start point

        # ensure there remains enough frame for training after random start point is set
        if avalible_len - start_frame < step_size:
            print( 'not enough image ' )
            pass
        else:
            # to enrich uncertainty, there are 50% chance that it need to perform 1 extra prediction  
            if( np.random.rand() > 0.5 ):
                exception = True
                step_size = step_size + 1
            else:
                exception = False

            # load every images needed in this batch 
            image_tensors = frame_batch_loader(start_frame, new_frame_paths, step_size, gray_scale = gray_scale_bol, size_index = size_idx).to(device)
                   
            for steps in range(0, step_size):

                # init_lstm_token is a token to tell model to initiallize lstm internal state or not, we should initilize internal state at start of each batch.
                if (steps == 0):
                    init_lstm_token = True
                else:
                    init_lstm_token = False

                #print("epoch", epochs, "steps", steps)
                # Clear the gradients, since PyTorch accumulates them
                start_time = time.time()
                optimizer.zero_grad()

                # load picture; step = pic num
                # test: input image
                # target: next image, prediction target image
                test, target = image_tensors[steps], image_tensors[steps+1]
                
                # FORWARD INPUT:
                # groundtruth image as input if steps < step
                # else: take previous output as input
                # eg. 5 GT input, 5 predictions ( step = 5 and predict = 5 )
                #     steps = 0: input groundtruth frame 0 -> output predict 1
                #     steps = 1: input groundtruth frame 1 -> output predict 2
                #     steps = 2: input groundtruth frame 2 -> output predict 3
                #     steps = 3: input groundtruth frame 3 -> output predict 4
                #     steps = 4: input groundtruth frame 4 -> output predict 5
                #   -------------------------------------------------------------- steps < step
                #     steps = 5: input predicted frame 5 -> output predict 6
                #     steps = 6: input predicted frame 6 -> output predict 7
                #     steps = 7: input predicted frame 7 -> output predict 8
                #     steps = 8: input predicted frame 8 -> output predict 9
    
                if steps <  step:
                    output = network.forward(test, init_lstm_token)
                    #if ( steps == step - 1 ):
                        #print('doing first prediction')
                else:
                    #print('doing prediction')
                    output = network.forward(previous_output, init_lstm_token)

                previous_output = output

                # Calculate loss
                # loss = critiria( Variable(output.long()),  Variable(target.long()))
                loss = critiria( output, target)

                # record loss in to csv
                loss_value =  float( loss.item() )
                string = 'epoch_' + str(epochs) + '_batch_' + str(batch) + '_step_' + str(steps)
                loss_list.append( [ string, loss_value ])

                # write training loss to tensorboard
                writer.add_scalar("train loss", loss.item(), epochs)
                
                # save img every 50 epochs ( 800 iterations ) 
                if save_img == True or float(loss_value) > 400:
                    if ( (epochs + 1) % 50 == 0) or ( epochs == 0 ) or ( (epochs+1) == ( start_epoch + epoch_num) ):
                        if steps % 1 == 0:
                            output_img = tensor_to_pic(output, normalize=False, gray_scale=gray_scale_bol, size_index = size_idx)
                            output_img_name = output_path + str(start_date) + '_E' + str(epochs) + '_B'+ str(batch).zfill(2) + '_S'+ str(steps).zfill(2) +'_output.jpg'
                            cv.imwrite(str(output_img_name), output_img)

                # Backward propagation
                loss.backward(retain_graph = True)
                
                # Update the gradients
                optimizer.step()

                # speed counter
                end_time = time.time()
                elapse_time = round((end_time - start_time), 2)

                # print memory used
                process = psutil.Process(os.getpid())

                if( epochs % 20 == 0 ):
                    print('epoch', epochs, 'batch', batch, 'step', steps, "loss:", loss, 'time used', elapse_time, 'sec')
                    print('used memory', round((int(process.memory_info().rss)/(1024*1024)), 2), 'MB' )
                    print("-------------------------------------")

                gc.collect()

                if cuda_gpu:
                    torch.cuda.empty_cache()
            
            # return origin step size if extra step is performed
            if exception == True:
               step_size = step_size - 1 
               exception = False
            # releae memory
            if cuda_gpu:
                torch.cuda.empty_cache()

    # log loss after each epoch
    write_csv_file( output_path + start_date +'_loss_record.csv', loss_list )
 
    # do validation (every 50 eopoch as default)
    if validation == True:
        print("==== validation ====\n\n")

        for batch in range(0, len(val_dir_list)):
            frame_paths = get_file_path( val_dir_list[batch] )  
            new_frame_paths = [ frame_paths[i] for i in range(0, len(frame_paths), skip_frame ) ]
            val_start = time.time()
            print(' ----batch{}----  '.format(batch))
            
            start_frame = 3
            image_tesnors = frame_batch_loader(start_frame, new_frame_paths, step_size, gray_scale = gray_scale_bol, size_index = size_idx).to(device)

            for steps in range(0, step_size):
                 test, target = image_tensors[steps], image_tensors[steps+1]

                 if steps == 0:
                     init_lstm_token = True
                 else:
                     init_lstm_token = False

                 if steps <  step:
                    output = network.forward(test, init_lstm_token)
                 else:
                    output = network.forward(previous_output, init_lstm_token)

                 previous_output = output
                 loss = critiria( output, target)
                 print('val: batch', batch, 'step', steps, "loss:", loss, '\n')
                 # write validation loss to tensorboard
                 writer.add_scalar("val loss", loss.item(), epochs)
                 writer.flush()

            val_end = time.time()
            print('time used:', round(( val_end - val_start ),2))

    # save model every 500 epochs
    if ( ( ( (epochs+1) % 500 ) == 0 ) or ((epochs+1) == ( start_epoch + epoch_num)) or ( (epochs+1)  == 1 ) ):
        path = output_path + start_date + 'epoch_' + str(epochs) +'_R_'+ str(step) + '_P_' + str(predict_frame_num) + '_size_idx_' + str(size_idx) +  '_R_Unet.pt'
        state = { 'epoch': epochs+1, 'state_dict': network.state_dict(), 'optimizer':optimizer.state_dict() }

        torch.save( state, path)
        print('save model to:', path)

    if cuda_gpu:
        torch.cuda.empty_cache()
