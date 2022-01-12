import tensorflow as tf

import cv2
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Convolution2D, MaxPooling2D, ZeroPadding2D, LeakyReLU, Conv2D, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from keras_flops import get_flops

from model.covariance_layer import *
from model.model_utils import *

DIR_TO_SAVE = "stuff/trained_models"
EXPERIMENT = "kernel_intializer"

def kernel_generation(subtract = 'cols'):
    model = Sequential()

    #Convolutional Block 1. N kernels per layer = [32,32]
    model.add(CovarianceLayer("32init", 7, 1, padding='SAME', subtract = subtract))
    model.add(CovarianceLayer("32",3,1, padding='SAME', subtract = subtract))
    model.add(MaxPooling2D((2,2)))

    #Convolutional Block 2. N kernels per layer = [64,64]
    model.add(CovarianceLayer("64", 3,1, padding='SAME', subtract = subtract))
    model.add(CovarianceLayer("64", 3,1, padding='SAME', subtract = subtract))
    model.add(MaxPooling2D((2,2)))                                                            

    #Convolutional Block 3. N kernels per layer = [128,128]
    model.add(CovarianceLayer("128",3,1, padding='SAME', subtract = subtract))
    model.add(CovarianceLayer("128",3,1, padding='SAME', subtract = subtract))

    return model 

def TrainingModel():
    lrelu = lambda x: tf.nn.leaky_relu(x)

    model = Sequential()
    model.add(Input(shape=(32,32,3)))

    #Convolutional Block 1. N kernels per layer = [32,32]
    model.add(Conv2D(32, (7,7) , activation=lrelu, padding='SAME'))
    model.add(BatchNormalization())                 
    model.add(Conv2D(32, (3,3) , activation=lrelu, padding='SAME'))
    model.add(BatchNormalization())   
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))                                                          

    #Convolutional Block 2. N kernels per layer = [64,64]
    model.add(Conv2D(64, (3,3) , activation=lrelu, padding='SAME'))
    model.add(BatchNormalization())  
    model.add(Conv2D(64, (3,3) , activation=lrelu, padding='SAME'))
    model.add(BatchNormalization())  
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.4))                                                            

    #Convolutional Block 3. N kernels per layer = [128,128]
    model.add(Conv2D(128, (3,3) , activation=lrelu, padding='SAME'))
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3,3) , activation=lrelu, padding='SAME'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3) , activation=lrelu, padding='SAME'))
    model.add(BatchNormalization())  
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5)) 

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.6))
    model.add(Dense(10, activation='softmax'))
    
    return model


def train(num_data_gen, num_data_train, num_data_test, dataset = 'cifar10', num_class = 10, freeze = False):
    #Define the names of the models
    model_name = f"{EXPERIMENT}_samples_{num_data_gen}"
    if freeze: model_name += "_freeze"
    model_dir = os.path.join(DIR_TO_SAVE, model_name)

    #Get the data to generate kernel
    data_to_generate, _ =  load_tf_data_custom('cifar10', 
                                    int(num_data_gen / num_class), 
                                    int(num_class / num_class), 
                                    num_class)

    print(f'Input Data Tensor Shape: {data_to_generate[0].shape} Type: {data_to_generate[0].dtype}')
    if data_to_generate[0].shape[0] != num_data_gen:
        print("ERROR tamaños de muestras no coinciden")
        return -1

    # Genererate the kernel
    gen_model = kernel_generation()
    result = gen_model(data_to_generate[0],training = True)
    print(f'[INFO] OUPUT SHAPE: {result.shape}')

    #Get the data tensor by class
    ds_train, ds_test =  load_tf_data_custom('cifar10', 
                                    int(num_data_train / num_class), 
                                    int(num_data_test / num_class), 
                                    num_class)

    # Load the generated kernel into Training Model
    full_model = TrainingModel()
    load_weights_in_full_tf(gen_model,full_model)

    del gen_model

    # Freeze layer
    if freeze:
        freeze_layers(full_model)

    #Training Procces
    epochs = 2
    opt = Adam(lr=0.001)
    full_model.compile(optimizer=opt, 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])

    # Define metrics to tensorboard
    log_dir = "stuff/logs/"+ model_name + "/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                            histogram_freq=10,
                                                            profile_batch=0)
        
    # Main Train
    full_model.fit(ds_train[0],ds_train[1], 
                    epochs=epochs,
                    batch_size=32, 
                    validation_data=(ds_test[0],ds_test[1]),
                    callbacks=[tensorboard_callback])

    # TODO: Save complete model
    full_model.save(model_dir + "_epochs_" + str(epochs))

    # Print trainable variables and FLOPS
    full_model.summary()

    flops = get_flops(full_model, batch_size=1)
    print(f"[INFO] FLOPS: {flops / 10 ** 6:.03} MFLOPS")
    # print("FLOPS: {:,} --- PARAMS: {:,}".format(flops.total_float_ops, params.total_parameters))


    return 0
