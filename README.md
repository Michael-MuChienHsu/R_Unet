# R_Unet
This is a on going project applying recurrent method upon U-net to perform video frame prediction </br>

Taking advantage of LSTM and U-net encode-decoder, we wish to be able in predicting next (n) frame(s). </br>
Currently using a 2 layer LSTM network as RNN network applying on latent feature of U net </br>

Our model currently on kth dataset outperforms Alex X. Lee et al. [1] upon SSIM and VGG cosine similarity evaluation matrics </br>
Also on kitti dataset outperforms Ruben V. et al. [2] upen SSIM and VGG cosine similarity evaluation matrics </br>

configuration: config.json </br>
parse configuration: class parse_arguement.py
training: train.py </br>
network model: R_Unet.py </br>

to train: python train.py config </br>

Current we are working on a better model using convolution lstm, name as runet_v2 </br>
Architecture of v1:
![alt_text](https://github.com/vagr8/R_Unet/blob/master/runet_v1.jpg)
</br>
Architecture of v2:
![alt_text](https://github.com/vagr8/R_Unet/blob/master/runet_v2.png)
</br>
</br>
[1] Stochastic Adversarial Video Prediction, CVPR 2018</br>
[2] High Fidelity Video Prediction with
Large Stochastic Recurrent Neural Networks, NeurIPS 2019</br>
</br>
Hsu Mu Chien, Watanabe Lab, Department of Fundamental Science and Engineering, Waseda University, All right reserved.
