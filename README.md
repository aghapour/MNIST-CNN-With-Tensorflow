# MNIST-CNN-With-Tensorflow

I get the first 9000 of dataset for training and 1000 dataset for testing of MNIST dataset. And down sample the original 28x28 data set to 14x14 and also augmented the 14x14 dataset to 14x14 dataset, another expression I used 3 type of different dataset to training, the number of all dataset for training is 36000. 
I repeated my training 20 time to get more accuracy, that called epoch. I declare epoch as a parameter of method you can test with variety of epoches to see the effect of epoch to increasing accuracy.
Here I used with epoch 20 to get more accuracy.

Data Augmentation
In my data augmentation I used translation, scales and rotation as I show in the code. I create augmented data equal size of the original data and in this case we had 9000 original dataset and 9000 augmented dataset I mixed them together and get 18000 dataset.
