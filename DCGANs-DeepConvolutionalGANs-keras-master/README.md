# DeepConvolutionalGANs-DCGANs-keras
This routine is implemented in Keras (Tensorflow as backend), it use the concept of DC-GANs to creat the convincing fake data compare with the import data. 

## Requirement:
* Python 3.5 or higher version
* Keras
* Tensorflow
* Pandas
* scikit-image

## Description:
* In this routine, it import the dataset is the identification photo of Asian's people.(6 female and 6 male). The image is shown like that:

![](https://raw.githubusercontent.com/q145492675/DeepConvolutionalGANs-DCGANs-keras/master/DCGANs_keras/GANs_dataset/girl_7.jpg)![](https://raw.githubusercontent.com/q145492675/DeepConvolutionalGANs-DCGANs-keras/master/DCGANs_keras/GANs_dataset/man_3.jpg)![](https://raw.githubusercontent.com/q145492675/DeepConvolutionalGANs-DCGANs-keras/master/DCGANs_keras/GANs_dataset/man_8.jpg)

## Usage:
* Download the whole routine.
* Run `test.py` 
* When it finishes training, the fake image created by the `Generator` will save in the folder of `result (the same path of test.py)`

## Applendix:
* It is the demo runtine, it still need to be improved.

* The training detail:
![](https://raw.githubusercontent.com/q145492675/DCGANs-DeepConvolutionalGANs-keras/master/DCGANs_keras/training_detail.png)

* Recent result: (After 25000 epochs)

![](https://raw.githubusercontent.com/q145492675/DeepConvolutionalGANs-DCGANs-keras/master/DCGANs_keras/result/predict/_0.jpg)![](https://raw.githubusercontent.com/q145492675/DeepConvolutionalGANs-DCGANs-keras/master/DCGANs_keras/result/predict/_5.jpg)![](https://github.com/q145492675/DeepConvolutionalGANs-DCGANs-keras/blob/master/DCGANs_keras/result/predict/_9.jpg)![](https://raw.githubusercontent.com/q145492675/DeepConvolutionalGANs-DCGANs-keras/master/DCGANs_keras/result/predict/_8.jpg)
