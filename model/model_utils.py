from matplotlib import pyplot
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import os
import cv2 as cv

from utils.utils import *

def make_cov_maxtrix(input_tensor,kernel_size, stride, subtract = None):
    nbatch ,height ,width ,channels = input_tensor.shape
    print('[INFO] Input Tensor in COV_MATRIX: {}'.format(input_tensor.shape))
    kernelWidth = kernel_size
    kernelHeight = kernel_size

    maxh = ((width-kernelWidth) // stride)+1
    maxv = ((height-kernelHeight) // stride)+1

    s_size = kernelHeight*kernelWidth*channels

    print("[INFO] S dim: s_size = ({},{}) maxh = {} maxv = {}".format(s_size,s_size,maxv,maxh))
    # print("IM HERE {} {}".format(input_tensor.shape, stride))

    S = tf.zeros((s_size, s_size),dtype=tf.float32)

    for batch in range(nbatch):
        #Create patches list
        patches = []

        for i in range(maxh):
            for j in range(maxv):
                
                #Temporal Patch
                tempP = tf.slice(input_tensor, [batch, j*stride, i*stride,0], [1, kernelHeight, kernelWidth, channels])

                #Flat the temporal patch
                tempP = tf.transpose(tempP)
                temFlat = tf.keras.backend.flatten(tempP)

                #Add the flatten patch to patches list
                patches.append(temFlat)

        #Create allpatches tensor
        allpatches = tf.concat([tf.expand_dims(t, 1) for t in patches], -1)

        if subtract == 'cols':
            #Subtract the mean by cols
            mean = tf.math.reduce_mean(allpatches,axis=0)
            allpatches = allpatches - mean
        elif subtract == 'rows':
            #Subtract the mean by rows
            mean = tf.math.reduce_mean(allpatches,axis=1)
            mean = tf.repeat(mean, repeats = [allpatches.shape[1]], axis=0)
            mean = tf.reshape(mean , shape = allpatches.shape)

            allpatches = allpatches - mean

        #Dot product between allpatches and transpose     
        dotprod = tf.tensordot(allpatches,tf.transpose(allpatches),1) / ( maxh * maxv )

        #Accumulate
        S = S + dotprod

    return (S/nbatch).numpy()

def generate_kernels(get_filters,input_tensor,kernel_0,kernel_size, stride, num_r, subtract = None):
    #Get S matrix
    S = make_cov_maxtrix(input_tensor,kernel_size,stride,subtract)

    #Print S
    #   print("S in Python")
    #   print_numpy(S)

    #Change the numpy to img_float
    cov_matrix, arr = numpy2cov_variance(S,input_tensor.shape,kernel_size, stride, num_r)
        
    #Get kernel block to c++
    kn_block = get_filters(cov_matrix)

    #Convert kernel to a list of numpys
    kernels_numpy = knblock2numpys(kn_block)[kernel_0:]
    kernels_numpy = [ kn.T for kn in kernels_numpy ] 

    #Convert the list of numpys to a tensor of tf
    kernels_np = np.array(kernels_numpy).T
    kernels_np = kernels_np.astype(np.float32)
    kernels_tensor = tf.convert_to_tensor(kernels_np)  
            
    return kernels_tensor

def store_weights(mirror_model, conv_model = None, dense_model = None):

    if not (conv_model is None):
        n_layer_conv_model = len(conv_model.layers)

        #Get the weights of the conv and save in the mirror
        for idx, layer in enumerate(mirror_model.layers[0:n_layer_conv_model]):
            # check for convolutional layer
            if 'conv' in layer.name:
                bs_layer = conv_model.layers[idx]
                weights = bs_layer.get_weights()
                # print(weights[0][:,:,0,0])
                layer.set_weights(weights)
                # print('-----------')

    if not (dense_model is None):
        n_layer_dense_model = len(dense_model.layers)

        #Get the weights of the dense and save in the mirror
        for idx, layer in enumerate(mirror_model.layers[-n_layer_dense_model:]):
            # check for convolutional layer
            if 'dense' in layer.name:
                bs_layer = dense_model.layers[idx]
                weights = bs_layer.get_weights()
                layer.set_weights(weights)

    return mirror_model

def summarize_diagnostics(history, model_name):
    route = f"results/{model_name}/train_plots"

    if not os.path.exists(route):
       os.makedirs(route)
	# plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig( route + '/' + model_name + '_plot.png')
    pyplot.close()

def get_trainX(ds_tf,num_train_batches):
  #ds_tf.batch(128)
  ds_tf = ds_tf.take(num_train_batches)
  ds_numpy = tfds.as_numpy(ds_tf)
  list_imgs = []
  for ex in ds_numpy:
    # `{'image': np.array(shape=(28, 28, 1)), 'labels': np.array(shape=())}`
    print(len(ex))
    print(ex[0].shape)
    #print(ex['image'].shape)
    list_imgs.append(ex[0])

  trainX = np.array(list_imgs)
  print(trainX.shape)
  return trainX

def load_data(dataset, num_batch_train=None, num_batch_test = None, resize = False, num_classes = 10):

    trainX, trainY = (None, None)

    if num_batch_train is not None:
        trainX, trainY = tfds.as_numpy(tfds.load(
            dataset,
            split='train', 
            batch_size=-1, 
            as_supervised=True,
        ))

    testX, testY = (None, None)
    
    if num_batch_test is not None:
        testX, testY = tfds.as_numpy(tfds.load(
            dataset,
            split='test', 
            batch_size=-1, 
            as_supervised=True,
        ))

    if num_batch_train is not None:
        trainX = trainX[:num_batch_train,:,:,:]
        trainY = trainY[:num_batch_train]

    if num_batch_test is not None:
        testX = testX[:num_batch_test,:,:,:]
        testY = testY[:num_batch_test]

    if num_batch_train is not None:
        trainX = tf.convert_to_tensor(trainX.astype(np.float32))/255
        trainY = tf.one_hot(trainY,num_classes)

    if num_batch_test is not None:
        testX = tf.convert_to_tensor(testX.astype(np.float32))/255
        testY = tf.one_hot(testY,num_classes)

    return trainX, trainY, testX, testY 
  
def load_tf_data(dataset, num_batch_train=None, num_batch_test = None, num_classes = 10):
    # Final function to export to the baseline pipe
    ds_train = None
    if num_batch_train is not None:
        ds_train, ds_info = tfds.load(
            dataset,
            split= 'train', 
            batch_size = -1,
            with_info = True,
        )
    ds_test = None
    if num_batch_test is not None:
        ds_test, ds_info = tfds.load(
                dataset,
                split= ['train', 'test'], 
                batch_size = -1,
                with_info = True,
            )   
    # print(type(num_batch_train))
    # we make an slice of the batch for the required images
    if num_batch_train is not None:
        ds_train['image'] = ds_train['image'][:num_batch_train]
        ds_train['label'] = ds_train['label'][:num_batch_train]
    if num_batch_test is not None:
        ds_test['image'] = ds_test['image'][:num_batch_test]
        ds_test['label'] = ds_test['label'][:num_batch_test]
      
    # We resize the images in batch to 320x320
    # and normalize the image between 0 and 1 and in float32
    # And finally we make the one hot tensor for labels
    if num_batch_train is not None:
        ds_train['image'] = tf.image.resize(ds_train['image'], [320,320])
        ds_train['image'] = tf.cast(ds_train['image'], tf.float32) / 255.
        ds_train['label'] = tf.one_hot(ds_train['label'],num_classes)
    if num_batch_test is not None:
        ds_test['image'] = tf.image.resize(ds_test['image'], [320,320])
        ds_test['image'] = tf.cast(ds_test['image'], tf.float32) / 255. 
        ds_test['label'] = tf.one_hot(ds_test['label'],num_classes)
    
    if ds_train is not None and ds_test is not None:    
        return ds_train['image'], ds_train['label'], ds_test['image'], ds_test['label']
    elif ds_train is not None:
        return ds_train['image'], ds_train['label'], None, None
    elif ds_test is not None:
        return None, None, ds_test['image'], ds_test['label']

def normalize_img(image, label):
  """Normalizes images: uint8 -> float32."""
  #if RESIZE: image = tf.image.resize(image, [32,32])
  return tf.cast(image, tf.float32) / 255., tf.one_hot(label, 10)

def normalize_cifar100(image, label):
  """Normalizes images: uint8 -> float32."""
  #if RESIZE: image = tf.image.resize(image, [32,32])
  return tf.cast(image, tf.float32) / 255., tf.one_hot(label, 100) 

def normalize_dtd(ds):
  """Normalizes images: uint8 -> float32."""
  ds['image'] = tf.image.resize(ds['image'], [32,32])
  return tf.cast(ds['image'], tf.float32) / 255., tf.one_hot(ds['label'], 47) 

def load_tf_data_no_batch(dataset, batch_size, as_supervised = True ):
    (ds_train, ds_test), ds_info = ((None, None), None)

    if as_supervised :
        # Final function to export to the baseline pipe
        (ds_train, ds_test), ds_info = tfds.load(
                dataset,
                split= ['train', 'test'],
                as_supervised=True, 
                # batch_size = -1,
                with_info = True,
            )
    else:
        # Final function to export to the baseline pipe
        (ds_train, ds_test), ds_info = tfds.load(
                dataset,
                split= ['train', 'test'],
                #as_supervised=True, 
                # batch_size = -1,
                with_info = True,
            )

    if dataset == 'cifar10':
        ds_train = ds_train.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if dataset == 'cifar100':
        ds_train = ds_train.map(
            normalize_cifar100, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if dataset == 'dtd':
        ds_train = ds_train.map(
            normalize_dtd, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    
    # ds_train = ds_train.cache()
    # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    #ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    if dataset == 'cifar10':
        ds_test = ds_test.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if dataset == 'cifar100':
        ds_test = ds_test.map(
            normalize_cifar100, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if dataset == 'dtd':
        ds_test = ds_test.map(
            normalize_dtd, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # ds_test = ds_test.cache()
    #ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds_train, ds_test


def load_tf_data_batch(dataset, batch_size, as_supervised = True, n_data_train = 10, n_data_test = 10, model_cov_tensor = None):
    (ds_train, ds_test), ds_info = ((None, None), None)

    if as_supervised :
        # Final function to export to the baseline pipe
        (ds_train, ds_test), ds_info = tfds.load(
                dataset,
                split = [f'train[:{n_data_train}]', f'test[:{n_data_test}]'],
                as_supervised=True, 
                # batch_size = -1,
                with_info = True,
            )
    else:
        # Final function to export to the baseline pipe
        (ds_train, ds_test), ds_info = tfds.load(
                dataset,
                split= ['train', 'test'],
                #as_supervised=True, 
                # batch_size = -1,
                with_info = True,
            )

    if dataset == 'cifar10':
        ds_train = ds_train.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if dataset == 'cifar100':
        ds_train = ds_train.map(
            normalize_cifar100, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if dataset == 'dtd':
        ds_train = ds_train.map(
            normalize_dtd, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if not (model_cov_tensor is None):
        ds_train = ds_train.map(
            model_cov_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # ds_train = ds_train.cache()
    # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    if dataset == 'cifar10':
        ds_test = ds_test.map(
            normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if dataset == 'cifar100':
        ds_test = ds_test.map(
            normalize_cifar100, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if dataset == 'dtd':
        ds_test = ds_test.map(
            normalize_dtd, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if not (model_cov_tensor is None):
        ds_test = ds_test.map(
            model_cov_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.batch(batch_size)
    # ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds_train, ds_test

def prep_pixels(train, test):
    train = train.astype(np.float32) / 255.0
    test = test.astype(np.float32) / 255.0

    return train, test

def load_weights(saved_model, conv_block):
    for idx, layer in enumerate(saved_model.layers):
        if 'conv' in layer.name:
            print(f'Load weight of layer {layer.name}')
            weights = layer.get_weights()
            conv_block.layers[idx].set_weights(weights)
    
    return conv_block

def load_weights_in_full_tf(saved_model,model_full_tf):
    #Get the indexs of the conv layers in the full tensor model
    idx_conv = []
    for idx, layer in enumerate(model_full_tf.layers):
        if 'conv' in layer.name:
            idx_conv.append(idx)

    #Load the weights in the full tensor model
    i = 0
    for layer in saved_model.layers:
        if 'covariance' in layer.name:
            #get the weights and bias
            weights = layer.get_weights()[0]
            bias = np.zeros((weights.shape[-1],),dtype=weights.dtype)
            print(f'[INFO] Load weight of layer {layer.name} in {model_full_tf.layers[idx_conv[i]].name} {weights.shape}')
            
            #set the new weights
            model_full_tf.layers[idx_conv[i]].set_weights([weights,bias])
            i += 1

def print_first_weights(model):
    for layer in model.layers:
        if 'conv' in layer.name:
            weights = layer.get_weights()[0]
            print(weights[:3,:3,0,0])
            break
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def load_tf_data_custom(dataset, num_img_per_class_train, num_img_per_class_test, num_class, shuffle = False, all_data = False, resize = False):
  (ds_train, ds_test) =tfds.as_numpy(tfds.load(
      dataset,
      split=['train', 'test'],
      # shuffle_files=True,
      as_supervised=True,
      batch_size = -1,
  ))

  if shuffle:
    ds_train = unison_shuffled_copies(ds_train[0],ds_train[1])
    ds_test = unison_shuffled_copies(ds_test[0],ds_test[1])
    print(f"First classes train: {ds_train[1][:10]}")
  
  #Channel Addition
  if 'mnist' in dataset:
    ds_train = (np.repeat(ds_train[0],3,axis=-1), ds_train[1])
    ds_test = (np.repeat(ds_test[0],3,axis=-1), ds_test[1])      

  train_batch = ds_train[1].shape[0]
  test_batch = ds_test[1].shape[0]

  ds_train_final, ds_test_final = (None, None)

  if (train_batch == int(num_img_per_class_train * num_class) or all_data):
    ds_train_final = (ds_train[0].astype(np.float32) /255.0, tf.one_hot(ds_train[1], num_class))

  if (test_batch == int(num_img_per_class_test * num_class) or all_data):
    ds_test_final = (ds_test[0].astype(np.float32) /255.0, tf.one_hot(ds_test[1], num_class))

  if (train_batch == int(num_img_per_class_train * num_class)) or (test_batch == int(num_img_per_class_test * num_class) or all_data):
    return ds_train_final, ds_test_final

  print(f"Train batch: {train_batch}, Test batch: {test_batch}")

  acc_train = [0]* num_class
  acc_test = [0]* num_class

  mask_train = [False] * train_batch
  mask_test = [False] * test_batch
  
  for idx in range(train_batch):
    if acc_train[ds_train[1][idx]] < num_img_per_class_train:
        acc_train[ds_train[1][idx]] += 1
        mask_train[idx] = True

    if sum(acc_test) >= num_img_per_class_train * num_class:
        break
      
  for idx in range(test_batch):
    if acc_test[ds_test[1][idx]] < num_img_per_class_test:
        acc_test[ds_test[1][idx]] += 1
        mask_test[idx] = True

    if sum(acc_test) >= num_img_per_class_test * num_class:
        break

  mask_nd_train = np.array(mask_train)
  mask_nd_test = np.array(mask_test)

  images_train = ds_train[0][mask_nd_train,:,:,:] / 255
  labels_train = ds_train[1][mask_nd_train]

  images_test = ds_test[0][mask_nd_test,:,:,:] / 255
  labels_test = ds_test[1][mask_nd_test]

  if resize:
      images_train = [cv.resize(image, (100, 100)) for image in images_train]
      images_test = [cv.resize(image, (100, 100)) for image in images_test]   

  ds_train_final = (tf.convert_to_tensor(images_train), tf.one_hot(labels_train, num_class))

  ds_test_final = (tf.convert_to_tensor(images_test), tf.one_hot(labels_test, num_class))

  return ds_train_final, ds_test_final    

def freeze_layers(model):
    indexes = []
    for idx, layer in enumerate(model.layers):
        if 'conv' in layer.name:
            indexes.append(idx)

    for i in indexes[:-1]:
        model.layers[i].trainable = False
        print("[INFO] Freezing layer: ", model.layers[i].name)
