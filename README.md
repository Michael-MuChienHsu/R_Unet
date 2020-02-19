# R_Unet
This is a on going project applying recurrent method upon U-net to perform pixel level video frame prediction </br>

# Brief introduction
Taking advantage of LSTM and U-net encode-decoder, we wish to be able in predicting next (n) frame(s). </br>
Currently using a 2 layer LSTM network (V1) or convoolution LSTM (V2) as RNN network applying on latent feature of U net </br>

Our model (V2) currently on kth dataset outperforms Alex X. Lee et al. [1] upon PSNR and VGG cosine similarity evaluation matrics </br>
Also on kitti dataset outperforms Ruben V. et al. [2] upen SSIM and VGG cosine similarity evaluation matrics </br>

# to use
* configuration: config.json </br>
* parse configuration: class parse_arguement.py </br>
* training file: train.py </br>
* V1 model: R_Unet.py </br>
* V2 model:  R_Unet.py </br>
```
to train: python train.py config </br>
```

# Our Architecture
Current we are working on a better model using convolution lstm, name as runet_v2 </br>
* Architecture of v1:
![alt_text](https://github.com/vagr8/R_Unet/blob/master/runet_v1.jpg)
</br>
* Architecture of v2:
![alt_text](https://github.com/vagr8/R_Unet/blob/master/runet_v2.png)
</br>
</br>

# References
[1] Stochastic Adversarial Video Prediction, CVPR 2018</br>
[2] High Fidelity Video Prediction with
Large Stochastic Recurrent Neural Networks, NeurIPS 2019</br>
</br>
Hsu Mu Chien, Watanabe Lab, Department of Fundamental Science and Engineering, Waseda University, All right reserved.
