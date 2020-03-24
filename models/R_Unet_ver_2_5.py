## Recurrent U-net, with LSTM
## default step = 6
## Future plan: multi-layer LSTM, conv LSTM, currently contains 2 layer LSTM

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import gc
from conv_lstm import ConvLSTM

# Down convolution layer
class Down_Layer(nn.Sequential):
    def __init__(self, ch_in, ch_out):
        super(Down_Layer, self).__init__()
        self.layer = self.define_layer( ch_in, ch_out )

    def define_layer(self, ch_in, ch_out):
        use_bias = True

        model = []
        model += [  nn.Conv2d( ch_in, ch_out, kernel_size=(3, 1), padding=1, bias=use_bias),
                    nn.Conv2d( ch_out, ch_out, kernel_size=(1,3), bias=use_bias),
                    nn.ReLU(True),
                    nn.Conv2d( ch_out, ch_out, kernel_size=(3, 1), padding=1, bias=use_bias),
                    nn.Conv2d( ch_out, ch_out, kernel_size=(1, 3), bias=use_bias),
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
        model += [  nn.Conv2d( self.ch_in, self.ch_out, kernel_size=(3, 1), padding=1, bias=use_bias),
                    nn.Conv2d( self.ch_out, self.ch_out, kernel_size=(1, 3), bias=use_bias),
                    nn.ReLU(True),
                    nn.Conv2d( self.ch_out, self.ch_out, kernel_size=(3, 1), padding=1, bias=use_bias),
                    nn.Conv2d( self.ch_out, self.ch_out, kernel_size=(1, 3), bias=use_bias),
                    nn.ReLU(True) ]

        return nn.Sequential(*model)

    def forward(self, x, resx):
        output = self.degradation( self.pad( self.upsample(x) ) )
        output = torch.cat((output, resx), dim = 1)

        output = self.layer(output)
        return output

class Up_Layer0(nn.Sequential):
    def __init__(self, ch_in, ch_out):
        super(Up_Layer0, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.layer1 = self.define_layer()
        self.layer2 = self.define_layer()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # add 0 padding on right and down to keep shape the same
        self.pad = nn.ConstantPad2d( (0, 1, 0, 1), 0 )
        self.degradation = nn.Conv2d( self.ch_in, self.ch_out, kernel_size=2 )

    def define_layer(self):
        use_bias = True
        pad = nn.ConstantPad2d( (0, 1, 0, 1), 0 )

        model = []
        model += [  nn.Conv2d( self.ch_in, self.ch_out, kernel_size=(3, 1), padding=1, bias=use_bias),
                    nn.Conv2d( self.ch_out, self.ch_out, kernel_size=(1, 3), bias=use_bias),
                    nn.ReLU(True),
                    nn.Conv2d( self.ch_out, self.ch_out, kernel_size=(3, 1), padding=1, bias=use_bias),
                    nn.Conv2d( self.ch_out, self.ch_out, kernel_size=(1, 3), bias=use_bias),
                    nn.ReLU(True) ]

        return nn.Sequential(*model)

    def forward(self, x, resx):
        # 1st conv box, up sample
        output = self.layer1( x )
        output = self.degradation( self.pad( self.upsample(x) ) )
               
        # concate output and res_x, 2nd conv_box
        output = torch.cat((output, resx), dim = 1)
        output = self.layer1(output)
        return output

class unet(nn.Module):
    def __init__(self, tot_frame_num = 100, step_ = 6, predict_ = 3 ,Gary_Scale = False, size_index = 256):
        print("gray scale:", Gary_Scale)
        super( unet, self ).__init__()
        if size_index != 256:
            self.resize_fraction = window_size = 256/size_index
        else:
            self.resize_fraction = 1

        cuda_gpu = torch.cuda.is_available()

        self.latent_feature = 0
        self.lstm_buf = []
        self.step = step_
        self.pred = predict_
        self.free_mem_counter = 0
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
       
        self.convlstm = ConvLSTM(input_channels=512, hidden_channels=[512, 512, 512], kernel_size=3, step=3,
                        effective_step=[2])

        if Gary_Scale == True:
            self.down1 = Down_Layer(1, 62)
        else:
            self.down1 = Down_Layer( 3, 62 )

        self.down2 = Down_Layer( 62, 120 )
        self.down3 = Down_Layer( 120, 224 )
        self.down4 = Down_Layer( 224, 384 )
        self.down5 = Down_Layer( 384, 512 )
        
        #self.up1 = Up_Layer0(1024, 512)
        self.up1 = Up_Layer(1024, 512)
        self.up2 = Up_Layer(512, 256)
        self.up3 = Up_Layer(256, 128)
        self.up4 = Up_Layer(128, 64)
        if Gary_Scale == True:
            self.up5 = nn.Conv2d( 64, 1, kernel_size = 1 )
        else:
            self.up5 = nn.Conv2d( 64, 3, kernel_size = 1 )
    
    def forward(self, x, free_token = False, test_model = False):
        '''
        self.free_token = free_token
        if ( self.free_token == True ):
            self.free_memory()
        '''
        # pop oldest buffer
        if( len(self.lstm_buf) >= self.step):   
            self.lstm_buf = self.lstm_buf[1:]
    
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

        latent_feature = x5.view(1, -1, int(16/self.resize_fraction), int(16/self.resize_fraction) )
        # add latest buffer
        # self.lstm_buf.append(latent_feature )
        if( test_model == True ):
            return latent_feature

        lstm_output =  Variable(self.convlstm(latent_feature)[0])

        if 'lstm_output' in locals():
            x5 = torch.cat((x5, lstm_output), dim = 1) 
            
            h = lstm_output.view(1, -1, x4.shape[2], x4.shape[3]) 
            #x4 = self.one_conv4(x4)
            x4 = torch.cat((x4, h), dim = 1) 
            x = self.up1( x5, x4 )

            h = lstm_output.view(1, -1, x3.shape[2], x3.shape[3]) 
            #x3 = self.one_conv5(x3)
            x3 = torch.cat((x3, h), dim = 1) 
            x = self.up2( x, x3 )

            h = lstm_output.view(1, -1, x2.shape[2], x2.shape[3]) 
            #x2 = self.one_conv6(x2)
            x2 = torch.cat((x2, h), dim = 1) 
            x = self.up3( x, x2 )

            h = lstm_output.view(1, -1, x1.shape[2], x1.shape[3]) 
            #x1 = self.one_conv7(x1)
            x1 = torch.cat((x1, h), dim = 1) 
            x = self.up4( x, x1 )

            x = F.relu(self.up5( x ))
        '''
        else:
            x5 = self.one_conv3( x5 )
        
            # up convolution 
            x = self.up1( x5, x4 )
            x = self.up2( x, x3 )
            x = self.up3( x, x2 )
            x = self.up4( x, x1 )
            x = F.relu(self.up5( x ))
        '''
        return x

    def free_memory(self):

        self.free_mem_counter = 0
