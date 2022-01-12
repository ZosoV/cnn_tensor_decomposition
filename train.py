import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from baselines.best_model import train

if __name__ == "__main__":
    # train(num_data_gen = 10, num_data_train = 50000, num_data_test = 10000)
    train(num_data_gen = 10, num_data_train = 50000, num_data_test = 10000, freeze = True)