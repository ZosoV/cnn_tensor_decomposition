import cv2
import numpy as np
from tensorflow.keras import Model
import os 

import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten


if not os.path.exists("results"):
    os.makedirs("results")

def write_kernel(input_numpy, str_layer, route):
    h, w, channels, nbatch = input_numpy.shape
    for i in range(nbatch):
        
        if (str_layer == '1'):
            img = input_numpy[:,:,:,i]
            img = cv2.resize(img,(320,320))
            img = ( ( img - np.amin(img) ) / (np.amax(img) - np.amin(img)) ) * 255

            img_name = f"{route}/kn_layer_{str_layer}_num_kernel_{i}.jpg"
            print(f"Storing {img_name}")
            cv2.imwrite(img_name,img)
 
        else:
            for j in range(channels):
                
                img = input_numpy[:,:,j,i]
                img = cv2.resize(img,(320,320))
                img = ( ( img - np.amin(img) ) / (np.amax(img) - np.amin(img)) ) * 255

                img_name = f"{route}/kn_layer_{str_layer}_num_kernel_{i}_ch_{j}.jpg"
                print(f"Storing {img_name}")
                # print("Kernel {} Max: {} Min: {}".format(i,np.amax(img),np.amin(img)))
                cv2.imwrite(img_name,img)

def write_kernels(model, model_name):
  route = f"results/{model_name}/kernels"

  if not os.path.exists(route):
    os.makedirs(route)

  count = 1
  for idx, layer in enumerate(model.layers):
    # check for convolutional layer
    if 'conv' in layer.name:
        weights = layer.get_weights()
        write_kernel(weights[0],str(count), route)
        count += 1

def write_feature_maps(model, input_tensor, model_name):
    route = f"results/{model_name}/feature_maps"

    if not os.path.exists(route):
        os.makedirs(route)

    ixs = []
    for idx, layer in enumerate(model.layers):
        # check for convolutional layer
        if 'conv' in layer.name:
            print(idx, layer.name, layer.output.shape)
            ixs.append(idx)

    outputs = [model.layers[i].output for i in ixs]
    model = Model(inputs=model.inputs, outputs=outputs)

    feature_maps = model.predict(input_tensor)

    for idx, fmps in enumerate(feature_maps):
        print(f'Shape fmps {fmps.shape}')
        write_feature_map(fmps, str(idx+1), route)

def write_feature_map(input_tensor, str_layer, route):
    nbatch, h, w, channels = input_tensor.shape

    for i in range(nbatch):
        for j in range(channels):           
            img = input_tensor[i,:,:,j] if isinstance(input_tensor, np.ndarray) else input_tensor.numpy()[i,:,:,j]

            # normalization
            img = ( ( img - np.amin(img) ) / (np.amax(img) - np.amin(img)) ) * 255
            
            img_name = f"{route}/fmp_layer_{str_layer}_num_kernel_{j}_batch_{i}.jpg"

            print(f'Storing {img_name}')
            cv2.imwrite(img_name,img)


def write_originals(input_tensor, model_name):

    route = f"results/{model_name}/originals"

    if not os.path.exists(route):
        os.makedirs(route)

    nbatch, h, w, channels = input_tensor.shape
    for i in range(nbatch):
        img = input_tensor[i,:,:,:] if isinstance(input_tensor, np.ndarray) else input_tensor.numpy()[i,:,:,:]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB )

        img_name = f"{route}/original_nbatch_{i}.jpg"
        print(f'Storing {img_name}')
        cv2.imwrite(img_name,img)



def plot_init_dist(saved_model, model_name, input_tensor):
    route = f"results/{model_name}/wa_plots/"

    if not os.path.exists(route):
        os.makedirs(route)

    w_layer = []
    layers_n = []

    for idx, layer in enumerate(saved_model.layers):
        if 'custom_conv' in layer.name:
            #get the weights and bias
            weights = layer.get_weights()[0]
            print(f'Load weight of layer {layer.name} idx:{idx} {weights.shape}')
            
            fweights = Flatten(weights).numpy()
            w_layer.append(fweights)
            layers_n.append(idx)

    outputs = [saved_model.layers[i].output for i in layers_n]
    model = Model(inputs=saved_model.inputs, outputs=outputs)

    feature_maps = model.predict(input_tensor)
    feature_maps = [Flatten(fm).numpy() for fm in feature_maps]
    print(feature_maps[0].shape)

    # for idx, fmps in enumerate(feature_maps):
    #     print(f'Shape fmps {fmps.shape}')
    #     write_feature_map(fmps, str(idx+1), route)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    axs[0].violinplot(w_layer, layers_n, points=200, widths=0.3,
                     showmeans=True, showextrema=True, showmedians=True)
    axs[0].set_title('Weights', fontsize=10)

    axs[1].violinplot(feature_maps, layers_n, points=200, widths=0.3,
                     showmeans=True, showextrema=True, showmedians=True)
    axs[1].set_title('Activation', fontsize=10)
    plt.savefig("test.jpg", bbox_inches='tight')