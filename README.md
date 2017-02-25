# CarND-Behavioral-Cloning-P3
Project #3 for Udacity Self-Driving Car Nanodegree: Behavioral Cloning

## Data

### Download data 
I used  data provided by Udacity: data.zip


### data_parser.py
I define a DataParser class that:
* Access the CSV file
* Stores the suffix of filename (e.g. 2016_12_11_22_52_25_418.jpg) in an array
* Stores the recorded steering angles in an array
* Provides left, center, and right images in batches (i.e. a few images at a time) as arrays

### model.py
Defines a generator that is called by model’s learning function to prepare chunks of data for learning. It grabs a random chunk of left, center, and right images and the corresponding steering angles. For each grouping of steering angle, 1 left, center, and right images, the generator does the following:
* Picks left, center, or right images with equal probability (33%)
* If the left or right images are picked, the steering angle is modified accordingly
* Image/steering angles are sometimes ignored - i.e. the higher the absolute value of the steering angle, the more likely we are to use the image for training
* We flip an image and steering angle with 50% probability
We then add the image and steering angle to the training batch

(carnd-term1) ppujari (master *) CarND-BehavioralCloning $ python model.py 
Using TensorFlow backend.
Running main in model.py
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 64, 64, 3)     12          lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 60, 60, 24)    1824        convolution2d_1[0][0]            
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 60, 60, 24)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 30, 30, 24)    0           elu_1[0][0]                      
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 30, 30, 24)    0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 26, 26, 36)    21636       dropout_1[0][0]                  
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 26, 26, 36)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 13, 13, 36)    0           elu_2[0][0]                      
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 13, 13, 36)    0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 11, 11, 48)    15600       dropout_2[0][0]                  
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 11, 11, 48)    0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 5, 5, 48)      0           elu_3[0][0]                      
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 5, 5, 48)      0           maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 3, 64)      27712       dropout_3[0][0]                  
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 3, 3, 64)      0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 1, 1, 64)      0           elu_4[0][0]                      
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 1, 1, 64)      0           maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 64)            0           dropout_4[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           6500        flatten_1[0][0]                  
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        elu_5[0][0]                      
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         elu_6[0][0]                      
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          elu_7[0][0]                      
====================================================================================================
Total params: 78,855
Trainable params: 78,855
Non-trainable params: 0
____________________________________________________________________________________________________
BehaviorCloner: train_model()...
Epoch 1/5
 1680/24108 [=>............................] - ETA: 191s - loss: 0.0733 
Epoch 2/5
24108/24108 [==============================] - 180s - loss: 0.0205     
Epoch 3/5
24108/24108 [==============================] - 180s - loss: 0.0177     
Epoch 4/5
24108/24108 [==============================] - 180s - loss: 0.0194     
Epoch 5/5
24108/24108 [==============================] - 181s - loss: 0.0154     
Accuracy =  0.00977379889837
... train_model() done
... main done
(carnd-term1) ppujari (master *) CarND-BehavioralCloning $ 


### Images
images are in images/


## Model Derivation
I began with the “Nvidia Model” from the paper - End to End Learning for Self-Driving Cars, April 25, 2016 (the paper is included in the repo as pdf). I then began adjusting the model and adding new features.

Features added include:
* Normalization
* Resizing of image
* Dropout
* Color Transform

Modifications include:
* Change number of ConvNet layers
* Changing ConvNet kernel sizes
* Changing boarder mode
* Changing RELU to ELU
* Adjusting ConvNet depths


## Simulator
Downloaded Simulator Available at:
* MacOS - https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip


## Model Architecture

### Normalization
I use a Keras lambda function to normalize the data between -1 to 1. Putting normalization in Keras allows the operation to be parallelized in GPUs and I don’t have to normalize manually when running the model during testing in the simulator in autonomous mode

### Color Transform
There is a 1x1,depth3 convolutional layer. It’s job is color space transformation. We could use OpenCV to do a color space transform, but it’s not clear what color space or spaces are most useful. Adding color transformation as a convolutional layer allows back-propagation to surmise the most useful color channels. Also, again since it’s in Keras, it is more efficient.

### Feature Extraction
There are 4 ConvNet layers. Each has:
* 2D Convolution
* ELU activation function
* Max Pooling
* Dropout

For the first two 2D Convolutions, we first do 5x5 to extract large features. Then the later two convolutions, we do 3x3 to extract groupings of features.

For the activation we use ELU instead of RELU, which was talked about in the lectures. With RELU some neurons can become inactive because the negative half of the RELU sends them to 0. ELU uses both positive and negative values and can train systems more consistently:
http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/

We use max pooling to bring down the dimensionality of the training and yield less data.

We use dropout to prevent overfitting to the specific images that the system is trained on. 


### Decision Making
First the data is flattened. Then 3 hidden layers are used, of sizes 100, 50, and 10 neurons. Each of these has a ELU activation function. Lastly, there is 1 output neuron.

## Validation
I validate the model by:
* Create a generator that only returns back center images and steering angles
* Run evaluate_generator() which runs feedforward on the images and compares them the steering angle, resulting in a loss value
