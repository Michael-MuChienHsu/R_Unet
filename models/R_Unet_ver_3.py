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

# Convolution unit
class conv_unit(nn.Sequential):
    def __init__(self, ch_in, ch_out):
        super(conv_unit, self).__init__()
        self.layer1 = self.define_layer1( ch_in, ch_out )
        self.layer2 = self.define_layer2( ch_in, ch_out )

        self.layer3 = self.define_layer1( ch_out, ch_out )
        self.layer4 = self.define_layer2( ch_out, ch_out )

        self.lamda1 = np.random.rand()
        self.lamda2 = np.random.rand()

    def define_layer1(self, ch_in, ch_out):
        use_bias = True

        model = []
        model += [  nn.Conv2d( ch_in, ch_out, kernel_size=(3, 1), padding=1, bias=use_bias),
                    nn.Conv2d( ch_out, ch_out, kernel_size=(1, 3), bias=use_bias),
                    nn.ReLU(True) ]

        return nn.Sequential(*model)

    def define_layer2(self, ch_in, ch_out):
        use_bias = True

        model = []
        model += [  nn.Conv2d( ch_in, ch_out, kernel_size=(5, 1), padding=2, bias=use_bias),
                    nn.Conv2d( ch_out, ch_out, kernel_size=(1, 5), bias=use_bias),
                    nn.ReLU(True) ]

        return nn.Sequential(*model)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        output = x1*(1-self.lamda1) + x2*(self.lamda1)

        x1 = self.layer3(output)
        x2 = self.layer4(output)
        output = x1*(1-self.lamda2) + x2*(self.lamda2)

        return output

# Up convolution layer
# input x and res_x
# upsamle(x) -> reduce_demention -> concatenate x and res_x -> up_conv_layer
class Up_Layer(nn.Sequential):
    def __init__(self, ch_in, ch_out):
        super(Up_Layer, self).__init__()
        #1st conv
        self.layer1 = self.define_layer1(ch_in, ch_out)
        self.layer2 = self.define_layer2(ch_in, ch_out)
        #2nd conv
        self.layer3 = self.define_layer1(ch_out, ch_out)
        self.layer4 = self.define_layer2(ch_out, ch_out)

        self.lamda1 = np.random.rand()
        self.lamda2 = np.random.rand()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # add 0 padding on right and down to keep shape the same
        self.pad = nn.ConstantPad2d( (0, 1, 0, 1), 0 )
        self.degradation = nn.Conv2d( ch_in, ch_out, kernel_size=2 )

    def define_layer1(self, ch_in, ch_out):
        use_bias = True

        model = []
        model += [  nn.Conv2d( ch_in, ch_out, kernel_size=(3, 1), padding=1, bias=use_bias),
                    nn.Conv2d( ch_out, ch_out, kernel_size=(1, 3), bias=use_bias),
                    nn.ReLU(True)]

        return nn.Sequential(*model)

    def define_layer2(self, ch_in, ch_out):
        use_bias = True

        model = []
        model += [  nn.Conv2d( ch_in, ch_out, kernel_size=(5, 1), padding=2, bias=use_bias),
                    nn.Conv2d( ch_out, ch_out, kernel_size=(1, 5), bias=use_bias),
                    nn.ReLU(True)]

        return nn.Sequential(*model)

    def forward(self, x, resx):
        output = self.degradation( self.pad( self.upsample(x) ) )
        output = torch.cat((output, resx), dim = 1)

        output1 = self.layer1(output) # 3conv
        output2 = self.layer2(output) # 5conv
        output =  (1- self.lamda1)*output1 + (self.lamda1)*output2

        output1 = self.layer3(output)
        output2 = self.layer4(output)
        output =  (1- self.lamda2)*output1 + (self.lamda2)*output2

        return output


# Up convolution layer
# input x and res_x
# upsamle(x) -> reduce_demention -> concatenate x and res_x -> up_conv_layer
class Up_Layer0(nn.Sequential):
    def __init__(self, ch_in, ch_out):
        super(Up_Layer0, self).__init__()
        #1st conv
        self.layer1 = self.define_layer1(ch_in, ch_out)
        self.layer2 = self.define_layer2(ch_in, ch_out)
        #2nd conv
        self.layer3 = self.define_layer1(ch_out, ch_out)
        self.layer4 = self.define_layer2(ch_out, ch_out)

        self.lamda1 = np.random.rand()
        self.lamda2 = np.random.rand()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # add 0 padding on right and down to keep shape the same
        self.pad = nn.ConstantPad2d( (0, 1, 0, 1), 0 )
        self.degradation = nn.Conv2d( ch_in, ch_out, kernel_size=2 )

    def define_layer1(self, ch_in, ch_out):
        use_bias = True

        model = []
        model += [  nn.Conv2d( ch_in, ch_out, kernel_size=(3, 1), padding=1, bias=use_bias),
                    nn.Conv2d( ch_out, ch_out, kernel_size=(1, 3), bias=use_bias),
                    nn.ReLU(True)]

        return nn.Sequential(*model)

    def define_layer2(self, ch_in, ch_out):
        use_bias = True

        model = []
        model += [  nn.Conv2d( ch_in, ch_out, kernel_size=(5, 1), padding=2, bias=use_bias),
                    nn.Conv2d( ch_out, ch_out, kernel_size=(1, 5), bias=use_bias),
                    nn.ReLU(True)]

        return nn.Sequential(*model)

    def forward(self, x, resx):
        output = self.degradation( self.pad( self.upsample(x) ) )
        output = torch.cat((output, resx), dim = 1)

        output1 = self.layer1(output) # 3conv
        output2 = self.layer2(output) # 5conv
        output =  (1- self.lamda1)*output1 + (self.lamda1)*output2

        output1 = self.layer3(output)
        output2 = self.layer4(output)
        output =  (1- self.lamda2)*output1 + (self.lamda2)*output2

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
            self.down1 = conv_unit(1, 62)
        else:
            self.down1 = conv_unit( 3, 62 )

        self.down2 = conv_unit(62, 120)
        self.down3 = conv_unit( 120, 224 )
        self.down4 = conv_unit( 224, 384 )
        self.down5 = conv_unit( 384, 512 )

        self.up1 = Up_Layer(1024, 512)
        self.up2 = Up_Layer(512, 256)
        self.up3 = Up_Layer(256, 128)
        self.up4 = Up_Layer(128, 64)

        if Gary_Scale == True:
            self.up5 = nn.Conv2d( 64, 1, kernel_size = 1 )
        else:
            self.up5 = nn.Conv2d( 64, 3, kernel_size = 1 )
    
    def forward(self, x, free_token, test_model = False):
        self.free_token = free_token
        if ( self.free_token == True ):
            self.free_memory()

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

        return x

    def free_memory(self):

        #self.convlstm.hidden_channels = self.convlstm.hidden_channels.detach()

        self.free_mem_counter = 0
