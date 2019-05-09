# Decomment if need to select GPUs to use
#import os
# The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0" 

from keras import backend as K
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.optimizers import RMSprop, SGD  
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Lambda, Flatten, Activation, Dropout
from keras.utils.generic_utils import get_custom_objects
from keras.datasets import mnist

import numpy as np
import pickle

#implementing swish, shiftet softplus and x + 0.5 tanh(x) in Keras using Tensorflow backend
def swish(x):
    return K.sigmoid(x) * K.identity(x)

def ssoftplus(x):
    return K.softplus(x) - np.log(2)

alpha = .5
def xplustanh(x):
    return K.identity(x) + alpha * K.tanh(x)


get_custom_objects().update({'swish': Activation(swish), 'xplustanh':Activation(xplustanh), 'ssoftplus':Activation(ssoftplus)})

#Parameters of the network
#
dim = 784 #For MNIST
#dim = 3072 #For CIFA10


# Loading et preprocessing MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
X_train = X_train / 255 
X_test = X_test / 255 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


# Loading et preprocessing CIFAR10
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
# num_pixels = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
# X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
# X_train = X_train / 255 
# X_test = X_test / 255 

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# num_classes = y_test.shape[1]


# Here we define a function 'simulation' to simplify the training for different parameters

def simulation(width, depth, act, sigma_w2, sigma_b2, q, opt, nb_simulations, nb_epochs, batch_size, batchnorm = False):

    def input_init_weights(shape, dtype=None):
        return K.variable(np.sqrt(sigma_w2/dim)*np.random.randn(shape[0], shape[1]))

    def init_weights(shape, dtype=None):
        return K.variable(np.sqrt(sigma_w2/width)*np.random.randn(shape[0], shape[1]))

    def init_bias(shape, dtype=None):
        return K.variable(np.sqrt(sigma_b2)*np.random.randn(shape[0]))
    
    x_train = np.sqrt(q) * X_train
    x_test = np.sqrt(q) * X_test


    def baseline_model():
        model = Sequential()
        model.add(Dense(width, input_shape=(dim,), kernel_initializer=input_init_weights, bias_initializer=init_bias,activation=act))
        for i in range(depth-1):
            model.add(Dense(width, kernel_initializer=init_weights, bias_initializer=init_bias,activation=act))
        model.add(Dense(num_classes, kernel_initializer=init_weights, bias_initializer=init_bias, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def baseline_model_bn():
        model = Sequential()
        model.add(Dense(width, input_shape=(dim,), kernel_initializer=input_init_weights, bias_initializer=init_bias))
        model.add(BatchNormalization())
        model.add(Activation(act))
        for i in range(depth-1):
            model.add(Dense(width, kernel_initializer=init_weights, bias_initializer=init_bias))
            model.add(BatchNormalization())
            model.add(Activation(act))
        model.add(Dense(num_classes, kernel_initializer=init_weights, bias_initializer=init_bias, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model
    
    
    # build the model
    res = [] #res will contain the test accuracies
    models = [baseline_model, baseline_model_bn]
    for i in range(nb_simulations):
        print('Step '+ str(i))
        # build the model
        Model = models[batchnorm]
        # Fit the model
        history = Model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nb_epochs, batch_size=batch_size, verbose=2)
        res.append(history.history['val_acc'])
    return res




## Tanh EOC
#  [(0, 1.0, 0.0),
#  (0.05, 1.1243396192538215, 0.15603045190402406),
#  (0.2, 1.3078679101523036, 0.5196867932931273),
#  (0.5, 1.55254475934761, 1.3259301630553542),
#  (1, 1.8575552209539357, 3.0437620971173187),
#  (5, 3.335472941814032, 34.63570399769325)]

width = 300
depth = 200
nb_epochs = 200
batchsize = 64
nb_runs = 10
actis = ['elu', 'tanh','relu']
params = [(0.1, 1.1775, 0.44), (0.2,1.3029, 0.5) , (0, np.sqrt(2), 1)] # values of (sigma_b, sigma_w, q) on the EOC . Use these values of EOC for ELU, TANH, RELU

LR_sgd = 0.0001
Learning_alg = SGD(lr = LR_sgd)
#LR_rms = 0.00001

for j in range(len(params)):
    Sigma_b, Sigma_w, q = params[j]
    print('Training for depth ' + str(depth) + 'with activation : ' + actis[j] + ' and sigma_b = ' + str(Sigma_b) )
    test = simulation(width,depth,actis[j], Sigma_w**2, Sigma_b**2, q, Learning_alg, nb_runs, nb_epochs, batchsize)
    name_file = 'res_'+actis[j]+'_' + str(width) + 'x'+str(depth)+'_' + str(nb_runs) +'simulations_SGD.pkl'
    with open(name_file, 'wb') as f: #saving the results
        pickle.dump([test], f)

