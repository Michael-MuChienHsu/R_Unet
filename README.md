# R_Unet
This project applies recurrent method upon U-net to perform pixel level video frame prediction. </br>
Part of our result is published at [IEEE GCCE 2020](https://ieeexplore.ieee.org/document/9292008) [pdf](https://www.ams.giti.waseda.ac.jp/data/pdf-files/2020_GCCE_hsu.pdf).</br>

# Brief introduction
Taking advantage of LSTM and U-net encode-decoder, we wish to be able in predicting next (n) frame(s). </br>
Currently using a 2 layer LSTM network (V1) or convolution LSTM (V2) as RNN network applying on latent feature of U net </br>
In our latest v4 model, we use convolutional LSTM in each level and take short cut in v2 out</br>

On the other hand, we are now using v4_mask model to train mask, image input and mask, image prediction output</br>
This model holds same structure as v4 but simply change output layer to output mask tensor. </br>

# Usage
* configuration: config.json </br>
* parse configuration: class parse_arguement.py </br>
* training file: train.py </br>
* V1 model: R_Unet_v1.py </br>
* V2 model:  R_Unet_ver_2.py </br>
* V4 model:  R_Unet_ver_4.py </br>
```
to train v1 model: python3 train.py config 
to train other model: python3 train_v2.py config 
```

# Our Model Architecture
Current we are working on a better model using convolution lstm, name as runet_v2 </br>
* model v1:</br>
![alt_text](https://github.com/vagr8/R_Unet/blob/master/runet_v1.jpg) </br> </br>

* model v2:</br>
![alt_text](https://github.com/vagr8/R_Unet/blob/master/pics/runet_v2.0_std.png) </br> </br>

* model v4:</br>
![alt_text](https://github.com/vagr8/R_Unet/blob/master/pics/runet_v4_a.png) 
</br>
</br>


# Some result
prediction:
 ![alt_text](https://github.com/vagr8/R_Unet/blob/master/pics/4_24000val.gif) 
Ground truth:
 ![alt_text](https://github.com/vagr8/R_Unet/blob/master/pics/4_24000val_gt.gif)
</br>
mask prediction
</br>
prediction:
 ![alt_text](https://github.com/vagr8/R_Unet/blob/master/pics/mask%206000%20gt.gif) 
Ground truth:
 ![alt_text](https://github.com/vagr8/R_Unet/blob/master/pics/mask%206000.gif)
<br>


# References
[1] Stochastic Adversarial Video Prediction, CVPR 2018</br>
[2] High Fidelity Video Prediction with
Large Stochastic Recurrent Neural Networks, NeurIPS 2019</br>
[3] [convLSTM](https://github.com/automan000/Convolutional_LSTM_PyTorch) - The convolution lstm framework used </br></br>
Hsu Mu Chien, Watanabe Lab, Department of Fundamental Science and Engineering, Waseda University, All right reserved.
