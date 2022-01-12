import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys

import tensorflow as tf
from utils.visualization import *
from model.model_utils import *

if __name__ == "__main__":

    model_directory = sys.argv[1]
    tmp = model_directory.split("/")[-1]
    model_name =  tmp if tmp != "" else model_directory.split("/")[-2]
    print(f"Load model: {model_name}")

    # TODO: Check reconstruction
    # It can be used to reconstruct the model identically.
    model = tf.keras.models.load_model(model_directory)

    #print_weights(model, 'conv')

    # Load data and predict results
    num_batch = 3
    dataset_name = 'cifar10'
    print(f"Load {num_batch} of {dataset_name} dataset.")
    trainX, trainY, testX, testY =  load_data(dataset_name, num_batch_train=1, num_batch_test=3, num_classes=10)
    # trainX, testX = prep_pixels(trainX, testX)

    results = model.predict(testX)



    # Write Results in jpg files
    write_originals(testX, model_name)
    write_kernels(model, model_name)
    write_feature_maps(model, testX, model_name)