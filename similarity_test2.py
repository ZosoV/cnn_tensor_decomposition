#Similary per Tensor of full kernels of different layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import seaborn as sns
import matplotlib.pyplot as plt
import os

from utils import utils

n_batches = [10, 100, 1000 , 2000, 3000, 4000]
n_kernel_per_layer = [32,32, 64,64, 128, 128]

baseline_name = "similarity_model30_cov_tensor"
model_name = "model30_conv_normal_cols_{}_no_leaky"
path_dir = "saved_models/" + model_name

len_batch = len(n_batches)

cov_per_layer = []
cor_per_layer = []
mse_per_layer = []

#Create the accumulators of cov per layers
for i in range(len(n_kernel_per_layer)):

    cov_acu = np.ones(shape=(len_batch,len_batch))
    cor_acu = np.ones(shape=(len_batch,len_batch))
    mse_acu = np.zeros(shape=(len_batch,len_batch))

    cov_per_layer.append(cov_acu)
    cor_per_layer.append(cor_acu)
    mse_per_layer.append(mse_acu)

layers_per_model = []

# Load the models to be probed
for batch_i in n_batches:
    model_dir = path_dir.format(batch_i)
    saved_model = tf.keras.models.load_model(model_dir)
    layers = saved_model.layers
    layers_per_model.append(layers)

# Get the metrics calculations
for i, bach_i in enumerate(n_batches):
    layers1 = layers_per_model[i]

    for j, batch_j in enumerate(n_batches):
        layers2 = layers_per_model[j]

        #if (batch_i == batch_j):
        #    continue
    
        layer_count = 0
        for k in range(len(layers1)):
            if 'conv' in layers1[k].name:

                weights_1 = tf.convert_to_tensor(layers1[k].get_weights()[0])
                weights_2 = tf.convert_to_tensor(layers2[k].get_weights()[0])

                cov_s = tfp.stats.covariance(weights_1, weights_2, sample_axis=[0,1,2,3], event_axis=None)
                cor_s = tfp.stats.correlation(weights_1, weights_2, sample_axis=[0,1,2,3], event_axis=None)                
                mse_s = utils.custom_mse(weights_1,weights_2, axis = (0,1,2,3))
                #print("Cov: {}".format(cov_s[:5]))
                #print("Cor: {}".format(cor_s[:5]))


                cov_per_layer[layer_count][i,j] = cov_s.numpy()
                cor_per_layer[layer_count][i,j] = cor_s.numpy()
                mse_per_layer[layer_count][i,j] = mse_s.numpy()

                layer_count +=1



def plot_and_save_confusion_matrix(data, metric, n_layer):

    route = f"results/{baseline_name}/full_tensor_kernel/"

    if not os.path.exists(route):
        os.makedirs(route)
    
    plt.figure()
    sns.heatmap(data, 
                xticklabels= n_batches,
                yticklabels= n_batches,
                annot=True,
                cmap='Blues')

    plt.xlabel("N Samples")
    plt.ylabel("N Samples")
    plt.title(f"{metric} per Model with N samples. \n Layer: {n_layer}, Full Tensor Kernel.")
    plt.savefig(route + f"{metric.lower()}_layer_{n_layer}.png")
    plt.close()

#Saving results
for idx_layer, _ in enumerate(n_kernel_per_layer):

    print(f"Saving result of Layer {idx_layer}, Full Kernel Tensor ")
    plot_and_save_confusion_matrix(mse_per_layer[idx_layer], "MSE", idx_layer)
    plot_and_save_confusion_matrix(cov_per_layer[idx_layer], "Covariance", idx_layer)
    plot_and_save_confusion_matrix(cor_per_layer[idx_layer], "Correlation", idx_layer)
