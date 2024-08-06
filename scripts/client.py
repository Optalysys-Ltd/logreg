import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import shutil
import sklearn
from sklearn.linear_model import LogisticRegression
import optalysys_explore.opt_api as opt_api
import time
import warnings
warnings.filterwarnings('ignore')

CLIENT_STORAGE_PATH = "output"
SERVER_STORAGE_PATH = "../input"
CRYPTO_FILE_PATH = os.path.join(CLIENT_STORAGE_PATH)
SECRET_DATA_PATH = os.path.join(CLIENT_STORAGE_PATH)
PUBLIC_DATA_PATH = os.path.join(SERVER_STORAGE_PATH)

def initialise_fhe(shape):
    """Initialises cryptographic parameters and materials such as keys and the server structure

    Args:
        shape:  Shape of input data in format [rows, columns] where the columns correspond to features
                and rows to individual transactions

    Returns:
        n_groups_bits: Number of groups of 4 bits in the message -> plaintext size limit
    """
    rows = shape[0]
    n_features = shape[1]
    # Free parameters
    n               = 512
    N               = 1024
    n_levels        = 4
    log_2_base      = 6
    Q               = 1179649
    noise_amplitude = 4.
    delta           = 64
    n_groups_bits   = 2
    size_data_chunk = rows

    # Pre-computed parameters
    sq_w = 523879
    inv_N = 1178497

    # Derived parameters
    q = N << 1

    # Structure with the parameters
    print("Initialising parameters")
    parameters = opt_api.OptParameters(n, N, n_levels, log_2_base, q, Q, sq_w, inv_N, delta)
    print("Initialising secret and bootstrapping keys...")
    opt_api.calculator_client_initialize(parameters, noise_amplitude)
    opt_api.calculator_server_initialize(sq_w, inv_N, log_2_base, n_levels, size_data_chunk)
    return n_groups_bits

def split_features(df):
    """Splits input full database into features, targets, and into test and training sets. Also
    applies undersampling technique to balance dataset as the raw datasets is very imbalanced

    Args:
        df: Pandas dataframe containing all input data for both training and testing

    Returns:
        X_undersampled_train: Training features
        X_undersampled_test: Testing features
        Y_undersampled_train: Training targets
        Y_undersampled_test: Testing targets
    """
    #Split into features and target variable
    print("Splitting data into features and targets...") 
    feature_names = df.columns[0]
    target_names = df.columns[1:]

    print("Balancing dataset through undersampling...")
    fraud_no = len(df[df.Class == 1])
    fraud_indices = df[df.Class == 1].index
    not_fraud_indices = df[df.Class == 0].index
    under_sample_indices = np.random.choice(not_fraud_indices, fraud_no, False)
    df_undersampled_pre = df.iloc[np.concatenate([fraud_indices, under_sample_indices]),:]

    # Duplicating dataset to create as many entries as we want #
    df_undersampled = df_undersampled_pre
    df_undersampled = pd.concat([df_undersampled_pre] * int(sys.argv[1]))

    print("Splitting datasets into training and testing data...")
    X_undersampled = df_undersampled.iloc[:,1:]
    Y_undersampled = df_undersampled.Class
    X_undersampled_train, X_undersampled_test, Y_undersampled_train, Y_undersampled_test = train_test_split(X_undersampled,
                                                                                                            Y_undersampled,
                                                                                                            test_size = 0.02)
    return X_undersampled_train, X_undersampled_test, Y_undersampled_train, Y_undersampled_test

def cleanup():
    """Helper function which cleans up directory structure in between runs
    """
    try:
        shutil.rmtree(CLIENT_STORAGE_PATH)
    except OSError:
        pass
    try:
        shutil.rmtree(SERVER_STORAGE_PATH)
    except OSError:
        pass
    os.mkdir(CLIENT_STORAGE_PATH)
    os.mkdir(SERVER_STORAGE_PATH)
    pass

def visualise(df):
    """Helper function which displays statistics of underlying dataset

    Args:
        df pandas.dataframe: Input dataset
    """
    print(f"\nLines and columns: {df.shape}\n")
    print('Not Fraud % ',round(df['Class'].value_counts()[0]/len(df)*100,2))
    print("\n")

    print('Fraud %    ',round(df['Class'].value_counts()[1]/len(df)*100,2))
    print("\n")

    print("Min value in each column:")
    print(df.min())
    print("\n")
    print("Max value in each column:")
    print(df.max())

def load_database(keep):
    """_summary_

    Args:
        keep list[int]: List containing indices of columns to keep from the dataset. This
        was precomputed by training a model and ignoring the least significant columns.

    Returns:
        df pandas.dataframe: Returns loaded dataset, with least significant columns removed
    """
    temp_df = pd.read_csv("creditcard.csv")
    df = temp_df.iloc[:, keep]
    print(df)
    visualise(df)
    return df

def train_model(X_undersampled_train, X_undersampled_test, Y_undersampled_train):
    """Initialises sklearn logistic regression model and trains it on the training data,
    also quantises the input training and testing data

    Args:
        X_undersampled_train (_type_): Input training features
        X_undersampled_test (_type_): Input testing features here to be quantised
        Y_undersampled_train (_type_): Input targets for training

    Returns:
        w list[float]: Weights extracted from trained logistic regression model
        binned_test np.array: Quantised testing input features - "binned" as in distributed
                              between equal sized bins
    """
    lr = LogisticRegression(max_iter=1000, n_jobs=-1, C=1, penalty='l2', fit_intercept=False)

    print("Quantising input features...")
    temp = sklearn.preprocessing.KBinsDiscretizer(n_bins = 256, encode='ordinal')
    binned = temp.fit_transform(X_undersampled_train)
    binned_test = temp.fit_transform(X_undersampled_test)
    lr.fit(binned, Y_undersampled_train)

    w = lr.coef_[0]
    print(f"Weights before scaling :\n {w}")
    print(type(binned))
    return w, binned_test

def quantise_weights(w):
    """Quantises the trained model weights to integer values between [-128, 128]
    needed due to accelerator hardware constraints

    Args:
        w list[float]: Trained model weights pre-quantisation

    Returns:
        qw list[int]: Quantised model weights
    """
    scaling_factor = (64 * (1/abs(min(w)) + 1/max(w)))
    print(f"Scaling factor: {scaling_factor}")
    qw = np.round((w * scaling_factor) + 128)       #5000 magic number -> scales weights to [-128, 128]
    print(f"Weights after scaling :\n {qw}")
    return list(qw)

def encrypt_feats(rows, n_groups_bits):
    """Encrypts the input testing features and generates and writes the client and server structures
    containing the secret and bootstrap keys respectively

    Args:
        rows list[list[int]]: List of rows containing the quantised test features for prediction
        n_groups_bits [int]: Number of groups of 4 bits in the message -> plaintext size limit 
    """
    print("Encrypt the messages...")
    print(f"Number of rows: {len(rows)}")
    for x in np.array(rows).T:
        for y in x:
            opt_api.calculator_client_encrypt(y, n_groups_bits)
    print("Save the ciphers...")
    opt_api.calculator_write_multiple_lwe(1, len(rows[0]) * len(rows), os.path.join(CRYPTO_FILE_PATH, "ciphers.dat"))

    print("Save the server structure...")
    opt_api.calculator_server_write(os.path.join(CRYPTO_FILE_PATH, "server_struct.dat"))

    print("Save the client structure...")
    opt_api.calculator_client_write(os.path.join(SECRET_DATA_PATH, "client_struct.dat"))
    print("Public data saved in " + CRYPTO_FILE_PATH)
    print("Secret client structure saved in " + SECRET_DATA_PATH + " (MUST NOT BE SHARED)")

def predict(X, weights):
    """Cleartext local prediction for accuracy comparison with FHE method

    Args:
        X list[list[int]]: List of cleartext rows containing input testing features
        weights list[int]: Quantised weights extracted from trained model

    Returns:
        y_predicted_cls [np.array]: Predicted classification over cleartext
    """
    linear_model = np.dot(X, np.array(weights[:-2]).T)
    y_predicted = (linear_model)
    temp = [128] * 17
    offsets = np.dot(X, temp)
    print(X.shape)
    count = 0
    y_predicted_cls = []
    for i in y_predicted:
        if i > offsets[count]:
            y_predicted_cls.append(1)
        else:
            y_predicted_cls.append(0)
        count += 1
    return np.array(y_predicted_cls)

if __name__ == '__main__':
    #Function to remove files from previous runs
    cleanup()
    print("Loading database...")
    print("Note: This database includes the Class column -> This is used for training and evaluation, never in classification.")
    print("\nDatabase loaded:\n")
    #Precomputed most significant columns
    print("Discarding least significant columns...")
    # -1 -> Class/Target variable
    keep_features = [-1, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18, 21, 28, 29]
    df = load_database(keep_features)
    X_undersampled_train, X_undersampled_test, Y_undersampled_train, Y_undersampled_test = split_features(df)
    print("\nTraining model...")
    w, binned_test = train_model(X_undersampled_train, X_undersampled_test, Y_undersampled_train)
    print("\nQuantising weights...")
    qw = quantise_weights(w)
    print("\nInitialise FHE structures...")
    n_groups_bits = initialise_fhe(binned_test.shape)
    print("\nEncrypting data...")
    temp_list = []
    for x in list(binned_test):
        #Reformat
        temp = []
        for y in x:
            temp.append(int(y))
        temp_list.append(temp)
    encrypt_feats(temp_list, n_groups_bits)
    print("\nWriting weights to files...")
    qw.append(binned_test.shape[1])
    qw.append(binned_test.shape[0])
    np.savetxt(os.path.join(CRYPTO_FILE_PATH,"weights.dat"), qw, fmt="%d")
    Y_undersampled_test.to_csv(os.path.join(SECRET_DATA_PATH, "data_target.csv"), index = False)
    print("\nSending data to server...")
    os.replace(CRYPTO_FILE_PATH, PUBLIC_DATA_PATH)
    print("\nQuantisation and encryption stage finished.")

    print("########################################")
    print("\nTesting non-FHE model")
    start = time.time()
    pred = predict(binned_test, qw)
    print(f"Time to predict : {time.time() - start}")
    print("\n\n######################################\n\n")
    print(sklearn.metrics.classification_report(Y_undersampled_test, pred))
    acc = sklearn.metrics.accuracy_score(Y_undersampled_test, pred)
    print(f"Accuracy score: {acc}")

    matches = (sum(a == b == 1 for a,b in zip(Y_undersampled_test, pred)))
    print(f"Number of correctly identified fraudulent transactions : {matches}")
    print(f"Realdata : {Counter(Y_undersampled_test)}")
    print(f"predictions: {Counter(pred)}")
    print(f"Size of test data: {np.round(binned_test.nbytes / 1024 / 1024, 3)}MB")
    print(f"Size of ciphers: {np.round(os.path.getsize(os.path.join(PUBLIC_DATA_PATH, 'ciphers.dat')) / 1024 / 1024, 3)}MB")
