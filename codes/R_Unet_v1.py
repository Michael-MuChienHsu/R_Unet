## Recurrent U-net, with LSTM
## default step = 6
## Future plan: multi-layer LSTM, conv LSTM, currently contains 2 layer LSTM

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import gc

# Down convolution layer
class Down_Layer(nn.Sequential):
    def __init__(self, ch_in, ch_out):
        super(Down_Layer, self).__init__()
        self.layer = self.define_layer( ch_in, ch_out )

    def define_layer(self, ch_in, ch_out):
        use_bias = True

        model = []
        model += [  nn.Conv2d( ch_in, ch_out, kernel_size=3, padding=1, bias=use_bias),
                    nn.ReLU(True),
                    nn.Conv2d( ch_out, ch_out, kernel_size=3, padding=1, bias=use_bias),
                    nn.ReLU(True) ]       

        return nn.Sequential(*model)

    def forward(self, x):
        return self.layer(x)

# Up convolution layer
# input x and res_x
# upsamle(x) -> reduce_demention -> concatenate x and res_x -> up_conv_layer
class Up_Layer(nn.Sequential):
    def __init__(self, ch_in, ch_out):
        super(Up_Layer, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.layer = self.define_layer( )

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # add 0 padding on right and down to keep shape the same
        self.pad = nn.ConstantPad2d( (0, 1, 0, 1), 0 )
        self.degradation = nn.Conv2d( self.ch_in, self.ch_out, kernel_size=2 )

    def define_layer(self):
        use_bias = True
        pad = nn.ConstantPad2d( (0, 1, 0, 1), 0 )

        model = []
        model += [  nn.Conv2d( self.ch_in, self.ch_out, kernel_size=3, padding=1, bias=use_bias),
                    nn.ReLU(True),
                    nn.Conv2d( self.ch_out, self.ch_out, kernel_size=3, padding=1, bias=use_bias),
                    nn.ReLU(True) ]       

        return nn.Sequential(*model)

    def forward(self, x, resx):
        output = self.degradation( self.pad( self.upsample(x) ) )
        output = torch.cat((output, resx), dim = 1)
        output = self.layer(output)
        return output

class recurrent_network(nn.Sequential):
    def __init__(self, fraction_index = 1):
        cuda_gpu = torch.cuda.is_available()
        self.resize_fraction = fraction_index
        super(recurrent_network, self).__init__()
        self.rnn = nn.LSTM(int(16/fraction_index), int(16/fraction_index) )
        if cuda_gpu:
            self.hidden1 = torch.zeros(1, int(16/self.resize_fraction), int(16/self.resize_fraction)).cuda()
            self.hidden2 = torch.zeros(1, int(16/self.resize_fraction), int(16/self.resize_fraction)).cuda()
        else:
            self.hidden1 = torch.zeros(1, int(16/self.resize_fraction), int(16/self.resize_fraction))
            self.hidden2 = torch.zeros(1, int(16/self.resize_fraction), int(16/self.resize_fraction))

    def forward(self, x):
        for i in x:
        # Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.
            out, (self.hidden1, self.hidden2) = self.rnn(i, (self.hidden1, self.hidden2) )

        return out

class recurrent_network_layer(nn.Sequential):
    def __init__(self, fraction_index = 1):
        super(recurrent_network_layer, self).__init__()
        cuda_gpu = torch.cuda.is_available()
        self.rnn = nn.LSTM(int(16/fraction_index), int(16/fraction_index) )
        self.resize_fraction = fraction_index
        self.free_mem_counter = 0
        if cuda_gpu:
            self.hidden1 = torch.zeros(1, int(16/self.resize_fraction), int(16/self.resize_fraction)).cuda()
            self.hidden2 = torch.zeros(1, int(16/self.resize_fraction), int(16/self.resize_fraction)).cuda()
        else:
            self.hidden1 = torch.zeros(1, int(16/self.resize_fraction), int(16/self.resize_fraction))
            self.hidden2 = torch.zeros(1, int(16/self.resize_fraction), int(16/self.resize_fraction))
        self.output_buffer = []

    def forward(self, x):
        self.init_buffer()
        for i in x:
        # Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.
            out, (self.hidden1, self.hidden2) = self.rnn(i, (self.hidden1, self.hidden2) )
            self.output_buffer.append(out)
            out.clone()
            del out
        
        return self.output_buffer
    
    def init_buffer(self):
        if len(self.output_buffer) > 0:
            self.output_buffer = []
    
class unet(nn.Module):
    def __init__(self, tot_frame_num = 100, length = 6, Gary_Scale = False, size_index = 256):
        print("gray scale:", Gary_Scale)
        super( unet, self ).__init__()
        if size_index != 256:
            self.resize_fraction = window_size = 256/size_index
        else:
            self.resize_fraction = 1

        cuda_gpu = torch.cuda.is_available()


        self.latent_feature = 0

        self.step = length
        self.free_mem_counter = 0
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.one_conv1 = nn.Conv2d( 1024, 512, kernel_size=1, bias=True)
        self.one_conv2 = nn.Conv2d( 1024, 512, kernel_size=1, bias=True)

        self.rnn = recurrent_network_layer( fraction_index = self.resize_fraction )
        self.rnn2 = recurrent_network( fraction_index = self.resize_fraction )

        if Gary_Scale == True:
            self.down1 = Down_Layer(1, 64)
        else:
            self.down1 = Down_Layer( 3, 64 )

        self.down2 = Down_Layer( 64, 128 )
        self.down3 = Down_Layer( 128, 256 )
        self.down4 = Down_Layer( 256, 512 )
        self.down5 = Down_Layer( 512, 1024 )

        self.up1 = Up_Layer(1024, 512)
        self.up2 = Up_Layer(512, 256)
        self.up3 = Up_Layer(256, 128)
        self.up4 = Up_Layer(128, 64)
        if Gary_Scale == True:
            self.up5 = nn.Conv2d( 64, 1, kernel_size = 1 )
        else:
            self.up5 = nn.Conv2d( 64, 3, kernel_size = 1 )
    
    def forward(self, x, buffer):
        self.lstm_buf = buffer.copy()
        self.free_mem_counter = self.free_mem_counter + 1
        #self.lstm_buf = []

        # down convolution
        x1 = self.down1(x)
        x2 = self.max_pool(x1)
        
        x2 = self.down2(x2)
        x3 = self.max_pool(x2)
        
        x3 = self.down3(x3)
        x4 = self.max_pool(x3)

        x4 = self.down4(x4)
        x5 = self.max_pool(x4)
        
        x5 = self.down5(x5)

        latent_feature = x5.view(-1, int(16/self.resize_fraction), int(16/self.resize_fraction) )

        self.lstm_buf.append(latent_feature )
        # print( 'lstm buffer len', len( self.lstm_buf ) )
        # LSTM unit
        if len( self.lstm_buf ) > 1 :
            lstm_output = self.rnn(self.lstm_buf)
            lstm_output = self.rnn2( lstm_output )
            lstm_output = lstm_output.view(1, 1024, int(16/self.resize_fraction), int(16/self.resize_fraction) )

        # use x5 to perform lstm
        if 'lstm_output' in locals():
            x6 = self.one_conv1(lstm_output)
            x5 = self.one_conv2(lstm_output)
            x5 = torch.cat((x5, x6), dim = 1)   
        
        # up convolution
        x = self.up1( x5, x4 )
        x = self.up2( x, x3 )
        x = self.up3( x, x2 )
        x = self.up4( x, x1 )
        x = F.relu(self.up5( x ))

        ## release var
        self.lstm_buf = []
        
        del x1, x2, x3, x4, x5
        if 'x6' in locals():
            del x6
        gc.collect()

        if self.free_mem_counter == 10 :
            self.free_memory()

        return x, latent_feature

    def free_memory(self):
        self.rnn.hidden1 = self.rnn.hidden1.detach()
        self.rnn.hidden2 = self.rnn.hidden2.detach()
        self.rnn2.hidden1 = self.rnn2.hidden1.detach()
        self.rnn2.hidden2 = self.rnn2.hidden2.detach()
        self.free_mem_counter = 0

