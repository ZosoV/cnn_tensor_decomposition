import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import seaborn as sns
import matplotlib.pyplot as plt
import os

from utils import utils


n_batches = [10, 100, 1000 , 2000] #, 2000, 2500, 3000]
n_kernel_per_layer = [32,32, 64,64, 128, 128]

baseline_name = "model30_similarity_cov_tensor_no_leaky"
model_name = "model30_conv_normal_cols_{}_no_leaky"
path_dir = "saved_models/" + model_name

len_batch = len(n_batches)

cov_per_layer = []
cor_per_layer = []
mse_per_layer = []

#Create the accumulators of cov per layers
for n_kernels in n_kernel_per_layer:

    cov_acu = np.ones(shape=(n_kernels,n_kernels,len_batch))
    cor_acu = np.ones(shape=(n_kernels,n_kernels,len_batch))
    mse_acu = np.zeros(shape=(n_kernels,n_kernels,len_batch))

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

    layer_count = 0
    for k in range(len(layers1)):
        if 'conv' in layers1[k].name:

            weights_1_0 = tf.convert_to_tensor(layers1[k].get_weights()[0])
            weights_1_1 = tf.convert_to_tensor(layers1[k].get_weights()[0])

            cov_s = tfp.stats.covariance(weights_1_0, weights_1_1, sample_axis=[0,1,2], event_axis=-1)
            cor_s = tfp.stats.correlation(weights_1_0, weights_1_1, sample_axis=[0,1,2], event_axis=-1)                
            mse_s = utils.custom_mse(weights_1_0,weights_1_1,axis=(0,1,2),event_axis=-1)
            #print("Cov: {}".format(cov_s[:5]))
            #print("Cor: {}".format(cor_s[:5]))

            cov_per_layer[layer_count][:,:,i] = cov_s.numpy()
            cor_per_layer[layer_count][:,:,i] = cor_s.numpy()
            mse_per_layer[layer_count][:,:,i] = mse_s.numpy()

            layer_count +=1

def plot_and_save_confusion_matrix(data, metric, n_layer, n_kernels, num_samples):

    route = f"results/{baseline_name}/in_layers/{metric.lower()}/"

    if not os.path.exists(route):
        os.makedirs(route)
    
    plt.figure(figsize=(12.8,9.6))
    sns.heatmap(data, 
                xticklabels= list(range(n_kernels)),
                yticklabels= list(range(n_kernels)),
                annot=False,
                cmap='Blues')

    plt.xlabel("N Samples")
    plt.ylabel("N Samples")
    plt.title(f"{metric} per Kernel with Model {num_samples} samples. \n Layer: {n_layer}.")
    plt.savefig(route + f"{metric.lower()}_layer_{n_layer}_samples_{num_samples}.png")
    plt.close()

#Saving results
for idx_layer, n_kernels in enumerate(n_kernel_per_layer):

    for idx, n_samples in enumerate(n_batches):

        print(f"Saving result of Layer {idx_layer} with model {n_samples} samples")
        plot_and_save_confusion_matrix(mse_per_layer[idx_layer][:,:,idx], "MSE", idx_layer, n_kernels, n_samples)
        plot_and_save_confusion_matrix(cov_per_layer[idx_layer][:,:,idx], "Covariance", idx_layer, n_kernels, n_samples)
        plot_and_save_confusion_matrix(cor_per_layer[idx_layer][:,:,idx], "Correlation", idx_layer, n_kernels, n_samples)
