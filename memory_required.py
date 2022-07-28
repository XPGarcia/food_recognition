from tensorflow import keras
from keras import backend as K
import numpy as np


def get_model_memory_usage(batch_size, model):
    features_mem = 0  # Initialize memory for features.
    float_bytes = 4.0  # Multiplication factor as all values we store would be float32.

    for layer in model.layers:

        out_shape = layer.output_shape

        if type(out_shape) is list:  # e.g. input layer which is a list
            out_shape = out_shape[0]
        else:
            out_shape = [out_shape[1], out_shape[2], out_shape[3]]

        # Multiply all shapes to get the total number per layer.
        single_layer_mem = 1
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s

        single_layer_mem_float = single_layer_mem * float_bytes  # Multiply by 4 bytes (float)
        single_layer_mem_MB = single_layer_mem_float / (1024 ** 2)  # Convert to MB

        # print("Memory for", out_shape, " layer in MB is:", single_layer_mem_MB)
        features_mem += single_layer_mem_MB  # Add to total feature memory count

    # Calculate Parameter memory
    trainable_wts = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_wts = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    parameter_mem_MB = ((trainable_wts + non_trainable_wts) * float_bytes) / (1024 ** 2)
    print("Memory for features in MB is:", features_mem * batch_size)
    print("Memory for parameters in MB is: %.2f" % parameter_mem_MB)

    total_memory_MB = (batch_size * features_mem) + parameter_mem_MB  # Same number of parameters. independent of batch size

    total_memory_GB = total_memory_MB / 1024

    print("Minimum memory required to work with this model is: %.2f" % total_memory_GB, "GB")
    print("_________________________________________")

    return total_memory_GB
