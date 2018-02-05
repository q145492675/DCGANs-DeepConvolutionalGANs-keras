# DeepConvolutionalGANs-DCGANs-keras
This routine is implemented in Keras (Tensorflow as backend), it use the concept of DC-GANs to creat the convincing fake data compare with the import data. GANs(Generative Adversarial Networks) is presented by Ian J.Goodfellows in 2014 `Goodfellow,I.et.al. In:Advances in neural information processing systems. pp.2672-2680 ` URL:https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf , 

It's an exciting generative network that train as an minmax and zero-sum game, the whole training process can be described as an example.'Generator', which can be described as a counterfeiter is try his best to create the fake money based on the real money. Meanwhile 'Discriminator', which can be described as a police, will try his best to verify which money is fake or real. Their training processes has extremely conflict between each other, and they will try their best to "overcome" the other one.

The whole training process can be shown as below:
![](https://raw.githubusercontent.com/q145492675/DCGANs-DeepConvolutionalGANs-keras/master/image1.png)

The structure of `Discriminator` and `Generator` in this routine (This is the original version. The import image size is 300*300; it might have some slight changes in the hyperparameters)
![](https://raw.githubusercontent.com/q145492675/DCGANs-DeepConvolutionalGANs-keras/master/image2.png)
![](https://raw.githubusercontent.com/q145492675/DCGANs-DeepConvolutionalGANs-keras/master/image3.png)

## Requirement:
* Python 3.5 or higher version
* Keras
* Tensorflow
* Pandas
* scikit-image

## Description:
* In this routine, the dataset it import is the identification photo of Asian's people.(6 female and 6 male). The image is shown like that:

![](https://raw.githubusercontent.com/q145492675/DCGANs-DeepConvolutionalGANs-keras/master/DCGANs-DeepConvolutionalGANs-keras-master/DCGANs_keras/GANs_dataset/girl_11.jpg)![](https://raw.githubusercontent.com/q145492675/DCGANs-DeepConvolutionalGANs-keras/master/DCGANs-DeepConvolutionalGANs-keras-master/DCGANs_keras/GANs_dataset/girl_7.jpg)![](https://raw.githubusercontent.com/q145492675/DCGANs-DeepConvolutionalGANs-keras/master/DCGANs-DeepConvolutionalGANs-keras-master/DCGANs_keras/GANs_dataset/man_2.jpg)



## Usage:
* Download the whole routine.
* Run `Master.py` 
* When it finishes training, the fake image created by the `Generator` will save in the folder of `result ` and name as `Generator.h5`


## Applendix:
* It is the demo runtine, it still need to be improved.

* The training detail:
![](https://raw.githubusercontent.com/q145492675/DCGANs-DeepConvolutionalGANs-keras/master/DCGANs-DeepConvolutionalGANs-keras-master/DCGANs_keras/training_detail.png)

* Recent result: (After 20000 epochs)

![](https://raw.githubusercontent.com/q145492675/DCGANs-DeepConvolutionalGANs-keras/master/DCGANs-DeepConvolutionalGANs-keras-master/DCGANs_keras/result/predict/_1.jpg)![](https://raw.githubusercontent.com/q145492675/DCGANs-DeepConvolutionalGANs-keras/master/DCGANs-DeepConvolutionalGANs-keras-master/DCGANs_keras/result/predict/_4.jpg)![](https://raw.githubusercontent.com/q145492675/DCGANs-DeepConvolutionalGANs-keras/master/DCGANs-DeepConvolutionalGANs-keras-master/DCGANs_keras/result/predict/_8.jpg)![](https://raw.githubusercontent.com/q145492675/DCGANs-DeepConvolutionalGANs-keras/master/DCGANs-DeepConvolutionalGANs-keras-master/DCGANs_keras/result/predict/_5.jpg)

## Which place can improve:
* More data might improve the performance, each epoch I will randomly choose `5 data` from the dataset, if we increase the number of data we import to GANs, it might improve the performance.
* The learning rate of the discriminator and the generator might improve the performance, but I think it's not very important (Have done a lot of works on it).
* The activation function of the discriminator and the generator.
* Improve the number of epochs.
* Other hyperparameters related to CNN.
