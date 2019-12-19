# R_Unet
This is a on going project applying recurrent method upon U-net to perform video frame prediction </br>

Taking advantage of LSTM and U-net encode-decoder, we wish to be able in predicting next (n) frame(s). </br>
Currently using a 2 layer LSTM network as RNN network </br>

for n input recurrently predict next 1 frame : train.py </br>
for n input predirct m output (future frames): train_mul.oy </br>

to train: python train.py config </br>
to train: python train_mul.py config </br>

Overview of network:
![alt_text](https://github.com/vagr8/R_Unet/blob/master/recurrent-u-net-architecture.png)
![alt text](https://github.com/vagr8/R_Unet/blob/master/laege.png)
</br>
</br>
Hsu Mu Chien, Watanabe Lab, Department of Fundamental Science and Engineering, Waseda University, All right reserved.
