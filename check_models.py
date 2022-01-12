import tensorflow as tf
from model import model_utils

num_batch_train = 10

cov_dir = f"model30_conv_normal_{num_batch_train}"
dense_dir = f"model30_dense_normal_{num_batch_train}_epochs_10"        
        
cov_model = tf.keras.models.load_model("saved_models/" + cov_dir)
dense_model = tf.keras.models.load_model("saved_models/" +dense_dir)

print("COV MODEL")
model_utils.print_first_weights(cov_model)

print("DENSE MODEL")
model_utils.print_first_weights(dense_model)

for layer in dense_model.layers:
    #print(layer.name)
    if 'sequential' in layer.name:
        for lay in layer.layers:
            print(lay.name)
            if 'conv' in lay.name:
                weights = lay.get_weights()[0]
                print(weights[:3,:3,0,0])
                break
