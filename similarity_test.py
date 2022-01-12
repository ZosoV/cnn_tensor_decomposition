#Similarity per kernel of different layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import seaborn as sns
import matplotlib.pyplot as plt
import os

from utils import utils

n_batches = [1, 5, 10 , 50, 100, 500, 1000] #1500] #, 2000, 2500, 3000]
n_kernel_per_layer = [32,64,64]

baseline_name = "model29_similarity_size_32x32"
model_name = "model29_conv_{}"
path_dir = "saved_models/" + model_name

len_batch = len(n_batches)

cov_per_layer = []
cor_per_layer = []
mse_per_layer = []

#Create the accumulators of cov per layers
for n_kernels in n_kernel_per_layer:

    cov_acu = np.ones(shape=(len_batch,len_batch,n_kernels))
    cor_acu = np.ones(shape=(len_batch,len_batch,n_kernels))
    mse_acu = np.zeros(shape=(len_batch,len_batch,n_kernels))

    cov_per_layer.append(cov_acu)
    cor_per_layer.append(cor_acu)
    mse_per_layer.append(mse_acu)

# Load the models to be probed

layers_per_model = []

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

                cov_s = tfp.stats.covariance(weights_1, weights_2, sample_axis=[0,1,2], event_axis=None)
                cor_s = tfp.stats.correlation(weights_1, weights_2, sample_axis=[0,1,2], event_axis=None)                
                mse_s = utils.custom_mse(weights_1,weights_2)
                #print("Cov: {}".format(cov_s[:5]))
                #print("Cor: {}".format(cor_s[:5]))


                cov_per_layer[layer_count][i,j,:] = cov_s.numpy()
                cor_per_layer[layer_count][i,j,:] = cor_s.numpy()
                mse_per_layer[layer_count][i,j,:] = mse_s.numpy()

                layer_count +=1



def plot_and_save_confusion_matrix(data, metric, n_layer, n_kernel):

    route = f"results/{baseline_name}/{metric.lower()}/"

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
    plt.title(f"{metric} per Model with N samples. \n Layer: {n_layer}, Kernel: {n_kernel}.")
    plt.savefig(route + f"{metric.lower()}_layer_{n_layer}_kernel_{n_kernel}.png")
    plt.close()

#Saving results
for idx_layer, n_kernels in enumerate(n_kernel_per_layer):

    for idx_kernel in range(n_kernels):

        print(f"Saving result of Layer {idx_layer}, Kernel {idx_kernel}")
        plot_and_save_confusion_matrix(mse_per_layer[idx_layer][:,:,idx_kernel], "MSE", idx_layer, idx_kernel)
        plot_and_save_confusion_matrix(cov_per_layer[idx_layer][:,:,idx_kernel], "Covariance", idx_layer, idx_kernel)
        plot_and_save_confusion_matrix(cor_per_layer[idx_layer][:,:,idx_kernel], "Correlation", idx_layer, idx_kernel)
