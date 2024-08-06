import optalysys_explore.opt_api as opt_api
import os
import pandas as pd
import sys
from collections import Counter

CRYPTO_FILE_PATH = os.path.join("client_storage","public_data")
SECRET_KEY_PATH = os.path.join("client_storage", "secret_data")
RECEIVE_PATH = os.path.join("output", sys.argv[1])
size_data_chunk = 20 #Number of rows sent for prediction -> change this if you change the number sent in client.py
n = 512

def decrypt(ciphers, n, size_data_chunk):
    """Decrypts outputs of FHE circuit computed in server.py 

    Args:
        ciphers string: Filepath to server outputs location
        n int: LWE key size
        size_data_chunk int: Number of rows sent for prediction
    Returns:
        results_raw list[int]: Output classifications from the encrypted model
    """
    print("Load the client structure...")
    opt_api.calculator_client_read(os.path.join(SECRET_KEY_PATH, "client_struct.dat"))
    print("Load the results...")
    index_cipher = opt_api.calculator_read_multiple_lwe(ciphers, n, 1)

    translate_res = {0: 1, 1: 0, 2: 1}
    results_raw = []
    for i in range(size_data_chunk):
        decrypted = translate_res[opt_api.calculator_client_decrypt(index_cipher + i)]
        results_raw.append(decrypted)
    return results_raw

def compare(results_raw):
    """Prints statistics to compare accuracy of prediction to real data

    Args:
        results_raw list[int]: Output classifications from the encrypted model
    """
    data_target = pd.read_csv(os.path.join(SECRET_KEY_PATH, "data_target.csv")) #Loads comparison data -> Real classification
    print("@############################@")
    print(f"Length of predictions: {len(results_raw)}")
    print("Predictions:")
    print(results_raw)
    print("Real Data:")
    targets = (data_target.Class.tolist())
    print(targets)
    print("@############################@")
    true_positives = sum(a == b == 1 for a,b in zip(targets , results_raw))
    false_positives = sum(a != b == 1 for a,b in zip(targets , results_raw))
    true_negatives = sum(a == b == 0 for a,b in zip(targets , results_raw))
    false_negatives = sum(a != b == 0 for a,b in zip(targets , results_raw))
    print(f"Number of correctly identified fraudulent transactions : {true_positives}")
    print(f"Number of incorrectly identified fraudulent transactions : {false_positives}")
    print(f"Number of correctly identified legitimate transactions : {true_negatives}")
    print(f"Number of incorrectly identified legitimate transactions : {false_negatives}")
    print("")
    print(f"Accuracy = {round((true_negatives + true_positives) / size_data_chunk, 2)}")
    print("")
    print(f"Real data : {Counter(data_target.Class)}")
    print(f"Predictions: {Counter(results_raw)}")
    print("@############################@")

if __name__ == '__main__':
    results_raw = decrypt(os.path.join(RECEIVE_PATH, "ciphers_res.dat"), 10, size_data_chunk)
    compare(results_raw)
    pass
