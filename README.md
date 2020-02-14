# R_Unet
This is a on going project applying recurrent method upon U-net to perform video frame prediction </br>

Taking advantage of LSTM and U-net encode-decoder, we wish to be able in predicting next (n) frame(s). </br>
Currently using a 2 layer LSTM network as RNN network applying on latent feature of U net </br>

configuration: config.json </br>
parse configuration: class parse_arguement.py
training: train.py </br>
network model: R_Unet.py </br>

to train: python train.py config </br>

Overview of network:
![alt_text](https://github.com/vagr8/R_Unet/blob/master/runet_v1.jpg)
</br>
</br>
Hsu Mu Chien, Watanabe Lab, Department of Fundamental Science and Engineering, Waseda University, All right reserved.
