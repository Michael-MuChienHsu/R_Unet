## Recurrent U-net, with LSTM
## D4 step = 6

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

    def define_layer(self):
        use_bias = True

        model = []
        model += [  nn.Conv2d( self.ch_in, self.ch_out, kernel_size=3, padding=1, bias=use_bias),
                    nn.ReLU(True),
                    nn.Conv2d( self.ch_out, self.ch_out, kernel_size=3, padding=1, bias=use_bias),
                    nn.ReLU(True) ]       

        return nn.Sequential(*model)

    def forward(self, x, resx):
        upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # add 0 padding on right and down to keep shape the same
        pad = nn.ConstantPad2d( (0, 1, 0, 1), 0 )
        degradation = nn.Conv2d( self.ch_in, self.ch_out, kernel_size=2 )

        x = upsample(x)
        x = pad( x )
        x = degradation(x)
        x = torch.cat((x, resx), dim = 1)
        out = self.layer(x)
        return out

class recurrent_network(nn.Sequential):
    def __init__(self, hidden_layers):
        super(recurrent_network, self).__init__()
        self.rnn = nn.LSTM(16, 16)
        self.hidden = hidden_layers

    def forward(self, x):
        for i in x:
        # Step through the sequence one element at a time.
        # after each step, hidden contains the hidden state.
            out, self.hidden = self.rnn(i, self.hidden)
        return out
    
class unet(nn.Module):
    def __init__(self, tot_frame_num = 100, length = 6):
        super( unet, self ).__init__()
        self.lstm_buf = []
        self.hidden = (torch.zeros(1, 16, 16),  # (hidden_layer num, second_dim, output channel)
                       torch.zeros(1, 16, 16))
        self.step = length
        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.one_conv1 = nn.Conv2d( 1024, 512, kernel_size=1, bias=True)
        self.one_conv2 = nn.Conv2d( 1024, 512, kernel_size=1, bias=True)

        self.rnn = recurrent_network( self.hidden )

        self.down1 = Down_Layer( 3, 64 )
        self.down2 = Down_Layer( 64, 128 )
        self.down3 = Down_Layer( 128, 256 )
        self.down4 = Down_Layer( 256, 512 )
        self.down5 = Down_Layer( 512, 1024 )

        self.up1 = Up_Layer(1024, 512)
        self.up2 = Up_Layer(512, 256)
        self.up3 = Up_Layer(256, 128)
        self.up4 = Up_Layer(128, 64)
        self.up5 = nn.Conv2d( 64, 3, kernel_size = 1 )
    
    def buffer_update(self, latent_feature):
        if len(self.lstm_buf) == self.step:
            for i in range(0, self.step-1):
                self.lstm_buf[i] = self.lstm_buf[i+1]
            self.lstm_buf[self.step-1] = latent_feature
        else:
            self.lstm_buf.append( latent_feature )

    def forward(self, x):
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
        self.buffer_update(latent_feature)

        if len( self.lstm_buf ) > 1 :
            lstm_output = self.rnn(self.lstm_buf)
            lstm_output = lstm_output.view(1, 1024, 16, 16)
        
        # use x5 to perform lstm
        if 'lstm_output' in locals():
            x6 = self.one_conv1(lstm_output)
            x5 = self.one_conv2(lstm_output)
            x5 = torch.cat((x5, x6), dim = 1)   
        
        x = self.up1( x5, x4 )
        x = self.up2( x, x3 )
        x = self.up3( x, x2 )
        x = self.up4( x, x1 )
        x = F.relu(self.up5( x ))

        return x

"""
## testing code
network = unet()
test_model = unet()
test = torch.randn( ( 1, 3, 256, 256 ) )
target = torch.tensor(torch.randn( ( 1, 3, 256, 256 ) ), dtype=torch.float)

import cv2 as cv

pic = cv.imread('./origami_single/001.jpg')
pic = torch.tensor(cv.resize(pic, (256,256), interpolation=cv.INTER_CUBIC ))
pic = pic.view(1, 3, 256, 256)
pic = torch.tensor(pic, dtype = torch.float)
#test = test.float() 
test = pic
optimizer = optim.SGD( network.parameters(), lr = 0.01 )
critiria = nn.MSELoss()
"""
def train():
    ##print(network)

    for epochs in (0, 5):
        for i in range(0, 100):
            print("epoch", epochs, "steps", i)
            # Clear the gradients, since PyTorch accumulates them
            optimizer.zero_grad()

            # Reshape and Forward propagation
            #test = unet_model.reshape(test)
            output = network.forward(test)

            # Calculate loss
            loss = critiria(output, target)

            # Backward propagation
            loss.backward(retain_graph=True)

            # Update the gradients
            optimizer.step()

            # save and evaluate every n epochs
            if ( (i > 0) and ( (i % 2) == 0)):
                # save model
                path = os.getcwd() + '/epoch_' + str(epochs) +"_step_" + str(i) + '_unet.pt'
                torch.save(network.state_dict(), path)

                # test model with train data
                print('loss at: epoch', epochs, 'step', i   , "loss", loss)

                #testing(path)

def testing(path):
    # load model
    try:
        test_model.load_state_dict(torch.load( path ))
        out = test_model(test)
        loss = critiria(out, target)
        ## print("Evaluation:", loss)
    except Exception as e:
        print("test err mag: ", str(e))

if __name__ == "__main__":
    train()
    
