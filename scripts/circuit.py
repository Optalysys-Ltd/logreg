import optalysys_explore.opt_api as opt_api
import os
import time
import subprocess
import sys

import numpy as np
from collections import deque

# Starting timer # 
start = time.time()

CLIENT_STORAGE = "output"
SERVER_STORAGE = "../input"
# ../input
# server_storage/public_data"
def load_weights(client_dir):
    """Loads server structure and ciphers from received client data

    Args:
        client_dir string: Location of directory containing data sent by client

    Returns:
        size_data_chunk int: Number of transactions to process -> number of rows
        weights list[int]: Quantised model weights
        n_features int: Number of features in model -> number of columns
    """
    weights = np.loadtxt(os.path.join(client_dir, "weights.dat"), dtype=int)
    print(weights)
    n_features = weights[-2]
    size_data_chunk = weights[-1]
    weights = weights[:-2]
    print(f"Size data chunk : {size_data_chunk}")
    return size_data_chunk, weights, n_features

def unpack_ciphers(n_features, size_data_chunk):
    """Unpacks LWE ciphers into RLWE ciphers for processing column by column (feature by feature)

    Args:
        n_features int: Number of features in model -> number of columns
        size_data_chunk int: Number of transactions to process -> number of rows
    """
    for i in range(n_features):
        print(f"Unpacking cipher: {i * size_data_chunk + 1} / {size_data_chunk * n_features}")
        opt_api.calculator_server_unpack(i * size_data_chunk + 1)

def homomorphic_dot_product(size_data_chunk, weights, n_groups_bits, n_features):
    """Calculates the products of ciphers and weights as well as defining queue which is used to calculate
    thresholds later

    Args:
        size_data_chunk int: Number of transactions to process -> number of rows
        weights list[int]: Quantised model weights
        n_groups_bits int: Number of groups of 4 bits in the message -> plaintext size limit 
        n_features int: Number of features in model -> number of columns

    Returns:
        queue_indices deque[int]: Double ended queue of cipher containing ciphers * weights
        queue_indices_thresholds deque[int]: Double ended queue of input ciphers to be used to calculate thresholds
    """
    queue_indices = deque()
    queue_indices_thresholds = deque()
    print("Products...")
    for i in range(0, n_features):
        print(f"Calculating product for feature no: {i}")
        cipher_index = opt_api.calculator_server_mul_plain(i * size_data_chunk + 1, [weights[i]] * size_data_chunk, 4 * n_groups_bits) #8 Bit output
        print(f"Saving at cipher index: {cipher_index}")
        queue_indices_thresholds.append(i * size_data_chunk + 1)
        queue_indices.append(cipher_index)
    return queue_indices, queue_indices_thresholds

def accumulation(queue_indices, str):
    """Summates the ciphers in a double ended queue

    Args:
        queue_indices deque[int]: Double ended queue containing indices of ciphers to be summed
        str string: Type of data used in print statement

    Returns:
        cipher_index int: Index of cipher containing result of summation
    """
    while (len(queue_indices) > 1):
        index_1 = queue_indices.pop()
        index_2 = queue_indices.pop()
        new_index = opt_api.calculator_server_add(index_1, index_2)
        queue_indices.appendleft(new_index)
        print(f"New {str} saved at: {new_index}")
    cipher_index = queue_indices.pop()
    return cipher_index

def run_FHE_circuit(client_dir):
    """Defines standard encryption parameters and loads server structures, sets up secondary
    parameters and loads ciphers then executes the FHE circuit


    Args:
        client_dir string: Filepath to directory containing client data
    """
    # Pre-agreed parameters
    n               = 512
    n_groups_bits   = 2

    print("Load the weights...")
    size_data_chunk, weights, n_features = load_weights(client_dir)

    # Structure with the parameters
    print("Load the server structure...")
    opt_api.calculator_server_read(os.path.join(client_dir, "server_struct.dat"))

    print("Load the ciphers...")
    index_cipher_input = opt_api.calculator_read_multiple_lwe(os.path.join(client_dir, "ciphers.dat"), n, n_groups_bits)

    print("Unpack the ciphers...")
    unpack_ciphers(n_features, size_data_chunk)

    print("Clear the LWE ciphers...")
    opt_api.calculator_lwe_clear()

    print("Homomorphic dot product...")
    queue_indices, queue_indices_thresholds = homomorphic_dot_product(size_data_chunk, weights, n_groups_bits, n_features)

    print("Accumulation...")
    cipher_index = accumulation(queue_indices, "cipher")
    threshold_index = accumulation(queue_indices_thresholds, "threshold")

    print("Calculating thresholds...")
    threshold_index = opt_api.calculator_server_mul_plain(threshold_index, [128] * size_data_chunk, 4 * n_groups_bits)

    print("Comparison...")
    cipher_index = opt_api.calculator_server_compare(cipher_index, threshold_index)

    print("Repack the ciphers...")
    lwe_index = opt_api.calculator_server_repack(cipher_index)

    print("Save the ciphers...")
    path_to_save = os.path.join(CLIENT_STORAGE, "ciphers_res.dat")
    opt_api.calculator_write_multiple_lwe(lwe_index, size_data_chunk, path_to_save)
    print(f"Saved to {path_to_save}")
    print(os.listdir(CLIENT_STORAGE))

def cleanup():
    """Helper function which cleans up directory structure in between runs
    """
    try:
        shutil.rmtree(SERVER_STORAGE)
    except OSError:
        pass
    try:
        shutil.rmtree(os.path.join(CLIENT_STORAGE, "received"))
    except OSError:
        pass
    os.mkdir(SERVER_STORAGE)

if __name__ == '__main__':
    # cleanup()
    print("Loading encrypted data...")
    run_FHE_circuit(SERVER_STORAGE)
    print("Sending back processed data...")
    try:
        os.mkdir(os.path.join(CLIENT_STORAGE, "received"))
    except OSError:
        pass

    print("Done sending back processed data.") 

    print(f"Time elapsed: {time.time() - start}")
