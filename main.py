'''
Created on Mar 20, 2017

@author:             Ahad Aghapour
@student number:    S011178
@email:            ahad.aghapour@ozu.edu.tr

'''

import os
import math
import struct
import numpy as np
import tensorflow as tf
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data



# Class Definition
class ahad_aghapour_training(object):
    
    network_type = None
    dataset_type = None
    MNIST_path = None
    
    # 28x28_dataset Variables    
    train_images_28x28 = None
    train_labels_28x28 = None
    test_images_28x28 = None
    test_labels_28x28 = None
    
    # 14x14_dataset Variables 
    train_images_14x14 = None
    train_labels_14x14 = None
    test_images_14x14 = None
    test_labels_14x14 = None
       
    # 14x14_augmented_dataset Variables
    train_images_aug_14x14 = None
    train_labels_aug_14x14 = None
    test_images_aug_14x14 = None
    test_labels_aug_14x14 = None
    
    
    # Class initializer 
    def __init__(self, network_type="network_1", dataset_type="dataset_28x28", MNIST_path=None):
        self.network_type = network_type
        self.dataset_type = dataset_type
        self.MNIST_path = MNIST_path
    
    
    # my method for downsampling    
    def downsample(self, a, shape2d):
        sh_end = shape2d[0], a.shape[-2] // shape2d[0], shape2d[1], a.shape[-1] // shape2d[1]
        a_tmp = a.reshape(a.shape[:-2] + sh_end)
        return a_tmp.mean(-1).mean(-2).reshape(-1, shape2d[0] * shape2d[1])
    
    # I create a methode to return batch
    def getNextBatch(self, train_images, train_labels, iteration, step_size):
        start = iteration * step_size
        end = (iteration + 1) * step_size
        return (train_images[start:end], train_labels[start:end])  
    
    # myAugmentor method
    def myAugmentation(self, dataset):
        warped_dataset = np.zeros([dataset.shape[0],dataset.shape[1]])
        for img_index in range(dataset.shape[0]):
            current_img = dataset[img_index,:]
            params = np.random.rand(3)
            tform=transform.SimilarityTransform(translation=(0, np.random.randint(-5,5) ), scale=params[1], rotation=params[2])
            warped=transform.warp(current_img.reshape(14,14), tform)
            warped_dataset[img_index,:] = warped.reshape(196)            
        return warped_dataset
       
        
    # load MNIST data from file if there is not exist extract it from tensorflow.examples.tutorials.mnist
    def createDatasets(self, numberOfTraining=9000, numberOfTesting=1000):
        # at the first it create the 28x28_dataset
        if self.MNIST_path == None:
            # Load data from tensorflow dataset
            mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            self.train_images_28x28 = mnist.train.images[0:numberOfTraining, :] 
            self.train_labels_28x28 = mnist.train.labels[0:numberOfTraining, :]
            self.test_images_28x28 = mnist.test.images[0:numberOfTesting, :]
            self.test_labels_28x28 = mnist.test.labels[0:numberOfTesting, :]
            print("Loaded {0} train data".format(self.train_images_28x28.shape)) 
            print("Loaded {0} test data".format(self.test_images_28x28.shape))  
            
            #
            self.train_dataset = mnist.train
        else:
            # Load data from MNIST file
            fname_img = os.path.join(self.MNIST_path, "t10k-images.idx3-ubyte")
            fname_lbl = os.path.join(self.MNIST_path, "t10k-labels.idx1-ubyte")            
            # Load everything in some numpy arrays
            with open(fname_lbl, 'rb') as flbl:  # labels
                magic, num = struct.unpack(">II", flbl.read(8))
                lbl = np.fromfile(flbl, dtype=np.int8)            
                print("Loaded {0} labels".format(lbl.shape))
                self.train_labels_28x28 = lbl[0:numberOfTraining]
                print('self.train_labels_28x28.shape: ', self.train_labels_28x28.shape)
                self.test_labels_28x28 = lbl[numberOfTraining:]
                print('self.test_labels_28x28.shape: ', self.test_labels_28x28.shape)
            with open(fname_img, 'rb') as fimg:  # images
                magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
                img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
                print("Loaded {} data".format(img.shape))
                self.train_images_28x28 = np.reshape(img, (10000, 784))
                self.test_images_28x28 = self.train_images_28x28[numberOfTraining:]
                self.train_images_28x28 = self.train_images_28x28[0:numberOfTraining]
                print('self.train_images_28x28.shape: ', self.train_images_28x28.shape)                
                print('self.test_images_28x28.shape: ', self.test_images_28x28.shape)
                
        
           
        # Create 14x14_dataset from down sampling of 28x28_dataset
        self.train_images_14x14 = self.downsample(self.train_images_28x28.reshape(-1, 28, 28), (14, 14))
        print('self.train_images_14x14.shape:  ', self.train_images_14x14.shape)
        self.train_labels_14x14 = self.train_labels_28x28
        self.test_images_14x14 = self.downsample(self.test_images_28x28.reshape(-1, 28, 28), (14, 14))
        print('self.test_images_14x14.shape:  ', self.test_images_14x14.shape)
        self.test_labels_14x14 = self.test_labels_28x28

        
        # Create 14x14_augmented_dataset from 14x14_dataset
        self.train_images_aug_14x14 = np.concatenate([self.train_images_14x14, self.myAugmentation(self.train_images_14x14)])
        print('train_images_aug_14x14:  ', self.train_images_aug_14x14.shape)
        self.train_labels_aug_14x14 = np.concatenate([self.train_labels_14x14, self.train_labels_14x14])
        self.test_images_aug_14x14 = self.test_images_14x14
        print('test_images_aug_14x14.shape:  ', self.test_images_aug_14x14.shape)
        self.test_labels_aug_14x14 = self.test_labels_14x14
        
        
     
    # Designed network_1    
    def network1(self, train_images, train_labels, test_images, test_labels, epoch=1):
        #
        sess = tf.InteractiveSession()
        
        #
        width = int(train_images.shape[1] ** (0.5))  # width of the image in pixels 
        height = width  # height of the image in pixels
        flat = width * height  # number of pixels in one image 
        class_output = 10  # number of possible classifications for the problem
        
        # Create place holders for inputs and outputs      
        x = tf.placeholder(tf.float32, shape=[None, flat])
        y_ = tf.placeholder(tf.float32, shape=[None, class_output])
        
        # Converting images of the data set to tensors
        # 28 pixels by 28 pixels or 14 pixels by 14 pixels and 1 channel (grayscale)
        x_image = tf.reshape(x, [-1, width, height, 1])  
        
        # Size of the filter/kernel: 3x3;
        # Input channels: 1 (greyscale);
        # 32 feature maps (here, 32 feature maps means 32 different filters are applied on each image. So, the output of convolution layer would be 28x28x32). In this step, we create a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
        W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))  # need 32 biases for 32 outputs
        
        # Defining a function to create convolutional layers. To creat convolutional layer, we use tf.nn.conv2d. It computes a 2-D convolution given 4-D input and filter tensors.
        # Stride size is 1
        convolve1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        
        # Apply the ReLU activation Function
        h_conv1 = tf.nn.relu(convolve1)
        
        # Use the max pooling operation, so the output of 28x28x1 would be 14x14x32
        # Kernel size: 2x2 (if the window is a 2x2 matrix, it would result in one output pixel)
        # Strides: dictates the sliding behavior of the kernel. In this case it will move 2 pixels everytime, thus not overlapping.
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max_pool_2x2
        
        # First layer completed        
        layer1 = h_pool1
        
        # Convolutional Layer 2
        # Weights and Biases of kernels
        # Filter/kernel: 5x5 (25 pixels) ; Input channels: 32 (from the 1st Conv layer, we had 32 feature maps); 64 output feature maps
        # Notice: here, the input is 14x14x32, the filter is 5x5x32 and the output of the convolutional layer would be 14x14x64
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))  # need 64 biases for 64 outputs
        
        # Convolve image with weight tensor and add biases.
        convolve2 = tf.nn.conv2d(layer1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
        
        # Apply the ReLU activation Function
        h_conv2 = tf.nn.relu(convolve2)        
        
        # Apply the max pooling      
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME')  # max_pool_2x2

        # Second layer completed    
        layer2 = h_pool2
        
        matrix_size = math.ceil(width/4)
        # Fully Connected Layer 3
        # Flattening Second Layer        
        layer2_matrix = tf.reshape(layer2, [-1, matrix_size * matrix_size * 64])        

        # Weights and Biases between layer 2 and 3      
        W_fc1 = tf.Variable(tf.truncated_normal([matrix_size * matrix_size * 64, 1024], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))  # need 1024 biases for 1024 outputs
        
        # Matrix Multiplication (applying weights and biases)      
        fcl3 = tf.matmul(layer2_matrix, W_fc1) + b_fc1        
        
        # Apply the ReLU activation Function      
        h_fc1 = tf.nn.relu(fcl3)        
        
        # Third layer completed        
        layer3 = h_fc1        
        
                
        # Optional phase for reducing overfitting - Dropout 3      
        keep_prob = tf.placeholder(tf.float32)
        layer3_drop = tf.nn.dropout(layer3, keep_prob)



        # Layer 4- Readout Layer (Softmax Layer)      
        # Type: Softmax, Fully Connected Layer.  
        # Weights and Biases
        W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))  # 1024 neurons
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))  # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]
                
        # Matrix Multiplication (applying weights and biases)        
        fcl4 = tf.matmul(layer3_drop, W_fc2) + b_fc2        
        
        # Apply the Softmax activation Function
        # softmax allows us to interpret the outputs of fcl4 as probabilities. So, y_conv is a tensor of probablities.
        y_conv = tf.nn.softmax(fcl4)        
        
        # layer4 finished        
        layer4 = y_conv


        
        # reduce_sum computes the sum of elements of (y_ * tf.log(layer4) across second dimension of the tensor, and reduce_mean computes the mean of all elements in the tensor..        
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(layer4), reduction_indices=[1]))
                
        # Define the optimizer        
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
                
        # Define prediction       
        correct_prediction = tf.equal(tf.argmax(layer4, 1), tf.argmax(y_, 1))
                
        # Define accuracy        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        # Run session
        sess.run(tf.global_variables_initializer())
        
        
        # Run train repeatedly for get more accurate
        for j in range(epoch):
            # Run 9000 or 180000 train sample (50*180=9000 or 50*360=18000) 
            for i in range(int(train_images.shape[0]/50)):
                # batch = train_dataset.next_batch(50)
                batch = self.getNextBatch(train_images, train_labels, i, 50)
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (i, float(train_accuracy)))

                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        
        
        # Evaluate the model
        networkaccuracy = accuracy.eval(feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})        
        return networkaccuracy
        
        
        # finish the session
        sess.close() 
        
        
        
            
        
    
     
    # Designed network_2    
    def network2(self, train_images, train_labels, test_images, test_labels, epoch=1):
        #
        sess = tf.InteractiveSession()
        
        #
        width = int(train_images.shape[1] ** (0.5))  # width of the image in pixels 
        height = width  # height of the image in pixels
        flat = width * height  # number of pixels in one image 
        class_output = 10  # number of possible classifications for the problem
        
        # Create place holders for inputs and outputs      
        x = tf.placeholder(tf.float32, shape=[None, flat])
        y_ = tf.placeholder(tf.float32, shape=[None, class_output])
        
        # Converting images of the data set to tensors
        # 28 pixels by 28 pixels or 14 pixels by 14 pixels and 1 channel (grayscale)
        x_image = tf.reshape(x, [-1, width, height, 1])  
        
        # Size of the filter/kernel: 5x5;
        # Input channels: 1 (greyscale);
        # 32 feature maps (here, 32 feature maps means 32 different filters are applied on each image. So, the output of convolution layer would be 28x28x32). In this step, we create a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
        W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))  # need 32 biases for 32 outputs
        
        # Defining a function to create convolutional layers. To creat convolutional layer, we use tf.nn.conv2d. It computes a 2-D convolution given 4-D input and filter tensors.
        convolve1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1
        
        # Apply the ReLU activation Function
        h_conv1 = tf.nn.relu(convolve1)
        
        # Use the max pooling operation, so the output would be 14x14x32
        # Kernel size: 2x2 (if the window is a 2x2 matrix, it would result in one output pixel)
        # Strides: dictates the sliding behaviour of the kernel. In this case it will move 2 pixels everytime, thus not overlapping.
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max_pool_2x2
        
        # First layer completed        
        layer1 = h_pool1
        
        # Convolutional Layer 2
        # Weights and Biases of kernels
        # Filter/kernel: 3x3 (9 pixels) ; Input channels: 32 (from the 1st Conv layer, we had 32 feature maps); 64 output feature maps
        # Notice: here, the input is 14x14x32, the filter is 3x3x32 and the output of the convolutional layer would be 14x14x64
        W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))  # need 64 biases for 64 outputs
        
        # Convolve image with weight tensor and add biases.
        convolve2 = tf.nn.conv2d(layer1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
        
        # Apply the ReLU activation Function
        h_conv2 = tf.nn.relu(convolve2)        
        
        # Apply the max pooling      
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')  # max_pool_2x2

        # Second layer completed    
        layer2 = h_pool2
        
        matrix_size = math.ceil(width/4)
        # Fully Connected Layer 3
        # Flattening Second Layer        
        layer2_matrix = tf.reshape(layer2, [-1, matrix_size * matrix_size * 64])        

        # Weights and Biases between layer 2 and 3      
        W_fc1 = tf.Variable(tf.truncated_normal([matrix_size * matrix_size * 64, 1024], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))  # need 1024 biases for 1024 outputs
        
        # Matrix Multiplication (applying weights and biases)      
        fcl3 = tf.matmul(layer2_matrix, W_fc1) + b_fc1        
        
        # Apply the ReLU activation Function      
        h_fc1 = tf.nn.relu(fcl3)        
        
        # Third layer completed        
        layer3 = h_fc1        
        
                
        # Optional phase for reducing overfitting - Dropout 3      
        keep_prob = tf.placeholder(tf.float32)
        layer3_drop = tf.nn.dropout(layer3, keep_prob)



        # Layer 4- Readout Layer (Softmax Layer)      
        # Type: Softmax, Fully Connected Layer.  
        # Weights and Biases
        W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))  # 1024 neurons
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))  # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]
                
        # Matrix Multiplication (applying weights and biases)        
        fcl4 = tf.matmul(layer3_drop, W_fc2) + b_fc2        
        
        # Apply the Softmax activation Function
        # softmax allows us to interpret the outputs of fcl4 as probabilities. So, y_conv is a tensor of probablities.
        y_conv = tf.nn.softmax(fcl4)        
        
        # layer4 finished        
        layer4 = y_conv


        
        # reduce_sum computes the sum of elements of (y_ * tf.log(layer4) across second dimension of the tensor, and reduce_mean computes the mean of all elements in the tensor..        
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(layer4), reduction_indices=[1]))
                
        # Define the optimizer        
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
                
        # Define prediction       
        correct_prediction = tf.equal(tf.argmax(layer4, 1), tf.argmax(y_, 1))
                
        # Define accuracy        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        # Run session
        sess.run(tf.global_variables_initializer())
        
        
        # Run train repeatedly for get more accurate
        for j in range(epoch):
            # Run 9000 train sample (50*180=9000) 
            for i in range(int(train_images.shape[0]/50)):
                # batch = train_dataset.next_batch(50)
                batch = self.getNextBatch(train_images, train_labels, i, 50)
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (i, float(train_accuracy)))
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        
        
        # Evaluate the model
        networkaccuracy = accuracy.eval(feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})        
        return networkaccuracy
        
        
        # finish the session
        sess.close() 
        
        
        

     
        
        
        



       

# my main block
if __name__ == "__main__":
    
    # method1 to extracting MNIST dataset is from file with directory path
    # t1 = ahad_aghapour_training(MNIST_path=os.getcwd())
    
    # method2 to extracting MNIST dataser is used from tensorflow samples dataset
    t1 = ahad_aghapour_training()
    
    # Loading 3 taype of datasets as assinment want to variables
    t1.createDatasets()
    
    # network1 accuracies
    print("28x28Dataset 9000 training sample and 1000 test sample, network1 Accuracy: ", t1.network1(t1.train_images_28x28, t1.train_labels_28x28, t1.test_images_28x28, t1.test_labels_28x28, epoch=20))
    print("14x14Dataset 9000 training sample and 1000 test sample, network1 Accuracy: ", t1.network1(t1.train_images_14x14, t1.train_labels_14x14, t1.test_images_14x14, t1.test_labels_14x14, epoch=20))    
    print("14x14Dataset 18000 training sample and 1000 test sample, network1 Accuracy: ", t1.network1(t1.train_images_aug_14x14, t1.train_labels_aug_14x14, t1.test_images_aug_14x14, t1.test_labels_aug_14x14, epoch=20))
    
    # network2 accuracies
    print("28x28Dataset 9000 training sample and 1000 test sample, network2 Accuracy: ", t1.network2(t1.train_images_28x28, t1.train_labels_28x28, t1.test_images_28x28, t1.test_labels_28x28, epoch=20))
    print("14x14Dataset 9000 training sample and 1000 test sample, network2 Accuracy: ", t1.network2(t1.train_images_14x14, t1.train_labels_14x14, t1.test_images_14x14, t1.test_labels_14x14, epoch=20))    
    print("14x14Dataset 18000 training sample and 1000 test sample, network2 Accuracy: ", t1.network2(t1.train_images_aug_14x14, t1.train_labels_aug_14x14, t1.test_images_aug_14x14, t1.test_labels_aug_14x14, epoch=20))
    
    