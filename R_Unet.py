## Recurrent U-net, with LSTM
## D4 step = 6

## Future plan: multi-layer LSTM, now 2 layer LSTM
## Conv_LSTM

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time

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
        x = self.upsample(x)
        x = self.pad( x )
        x = self.degradation(x)
        x = torch.cat((x, resx), dim = 1)
        x = self.layer(x)
        return x

class recurrent_network(nn.Sequential):
    def __init__(self, hidden_layer1, hidden_layer2, use_buffer = False):
        super(recurrent_network, self).__init__()
        self.use_buffer = use_buffer
        self.rnn = nn.LSTM(16, 16)
        self.hidden1 = hidden_layer1
        self.hidden2 = hidden_layer2
        if use_buffer == True:
            self.output_buffer = []

    def forward(self, x):
        if self.use_buffer == False:
            for i in x:
            # Step through the sequence one element at a time.
            # after each step, hidden contains the hidden state.
                out, (self.hidden1, self.hidden2) = self.rnn(i, (self.hidden1, self.hidden2) )
            
            return out

        else:
            self.output_buffer = []
            for i in x:
            # Step through the sequence one element at a time.
            # after each step, hidden contains the hidden state.
                out, (self.hidden1, self.hidden2) = self.rnn(i, (self.hidden1, self.hidden2))
                self.output_buffer.append(out)
            return self.output_buffer
    
class unet(nn.Module):
    def __init__(self, tot_frame_num = 100, length = 6, Gary_Scale = False):
        super( unet, self ).__init__()
        cuda_gpu = torch.cuda.is_available()

        if cuda_gpu:
            self.hidden11 = torch.zeros(1, 16, 16).cuda  # (hidden_layer num, second_dim, output channel)
            self.hidden12 = torch.zeros(1, 16, 16).cuda
            self.hidden21 = torch.zeros(1, 16, 16).cuda  # (hidden_layer num, second_dim, output channel)
            self.hidden22 = torch.zeros(1, 16, 16).cuda
        else:
            self.hidden11 = torch.zeros(1, 16, 16)  # (hidden_layer num, second_dim, output channel)
            self.hidden12 =  torch.zeros(1, 16, 16)
            self.hidden21 = torch.zeros(1, 16, 16)  # (hidden_layer num, second_dim, output channel)
            self.hidden22 = torch.zeros(1, 16, 16)

        self.step = length
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.one_conv1 = nn.Conv2d( 1024, 512, kernel_size=1, bias=True)
        self.one_conv2 = nn.Conv2d( 1024, 512, kernel_size=1, bias=True)

        self.rnn = recurrent_network( self.hidden11, self.hidden12, use_buffer = True )
        self.rnn2 = recurrent_network( self.hidden21, self.hidden22 )

        self.down1 = Down_Layer( 3, 64 )
        if Gary_Scale == True:
            self.down1 = Down_Layer(1, 64)

        self.down2 = Down_Layer( 64, 128 )
        self.down3 = Down_Layer( 128, 256 )
        self.down4 = Down_Layer( 256, 512 )
        self.down5 = Down_Layer( 512, 1024 )

        self.up1 = Up_Layer(1024, 512)
        self.up2 = Up_Layer(512, 256)
        self.up3 = Up_Layer(256, 128)
        self.up4 = Up_Layer(128, 64)
        self.up5 = nn.Conv2d( 64, 3, kernel_size = 1 )
        if Gary_Scale == True:
            self.up5 = nn.Conv2d( 64, 1, kernel_size = 1 )
    '''
    ## move to utils
    def buffer_update(self, latent_feature):
        if len(self.lstm_buf) == self.step:
            for i in range(0, self.step-1):
                self.lstm_buf[i] = self.lstm_buf[i+1]
            self.lstm_buf[self.step-1] = latent_feature
        else:
            self.lstm_buf.append( latent_feature )
    '''
    def forward(self, x, buffer):
        self.lstm_buf = buffer.copy()

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

        latent_feature = x5.view(-1, 16, 16)
        self.lstm_buf.append( latent_feature )
        print( 'lstm buffer len', len( self.lstm_buf ) )
        # LSTM unit
        if len( self.lstm_buf ) > 1 :
            lstm_output = self.rnn(self.lstm_buf)
            lstm_output = self.rnn2( lstm_output )
            lstm_output = lstm_output.view(1, 1024, 16, 16)
        
        # use x5 to perform lstm
        if 'lstm_output' in locals():
            x6 = self.one_conv1(lstm_output)
            x5 = self.one_conv2(lstm_output)
            x5 = torch.cat((x5, x6), dim = 1)   
            del lstm_output
        
        # up convolution
        x = self.up1( x5, x4 )
        x = self.up2( x, x3 )
        x = self.up3( x, x2 )
        x = self.up4( x, x1 )
        x = F.relu(self.up5( x ))

        ## release var
        self.lstm_buf = []

        return x, latent_feature
