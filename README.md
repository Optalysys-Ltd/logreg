# Logistic regression over encrypted data
This is a demonstration of a logistic regression model operating over encrypted data, written in the Python interface of the [Optalysys beta API](https://optalysys.gitbook.io/optalysys-accelerator-documentation/QQvmmApy5f2RR4eiSHLZ/) (Click link to find out more).
The model is trained on a [dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)[^1][^2] of legitimate and fraudulent credit card transactions made by European cardholders in 2013.

Our model was able to achieve ~95% accuracy in fraud detection under FHE with a negligible drop from conventional plaintext accuracy arising from quantisation.

[^1]:Contains information from the "Credit Card Fraud Detection" database, which is made available here under the [Open Database License](https://opendatacommons.org/licenses/odbl/1-0/) (ODbL).
[^2]: This is not included in the repository and requires downloading as described in the setup section.

## Fraud Detection Demo
The envisioned workflow for a real world application is as follows:

![Cloud encrypted fraud detection](Architecture_Diagram.png)
1. The client (e.g. the bank) has a list of confidential transactions that it wants to check for fraudulent activity.
2. The client coverts the data into a format accepted by our hardware accelerator using quantisation techniques.
3. The data is then encrypted using secure parameters under TFHE. Alongside this, a server key is generated.
4. The client sends the encrypted transaction data to a third party (potentially another bank to allow collaborative fraud checking)  which then sends the weights and biases along with the clients data to the Optalysys cloud accelerator.
5. Optalysys cloud accelerator performs homomorphic operations to classify encrypted transactions as fraudulent or not and returns an encrypted answer to the client.
6. Client decrypts result of classification.

### Demo Structure
The workflow presented here is different for ease of demonstration, but this could easily be adapted to a real world scenario.
In the demo, the model is trained during runtime by the client and the weights are sent to the server, this is done to simplify comparison between predicion on cipher/plain-text.
For simplicity's sake, this demo contains no net-code but simulates an end-to-end encrypted flow through the splitting of server and client-side code.

### Quantisation
The architecture of our accelerator and the constraints of TFHE require integer inputs of bounded size. In this demo, we quantise the inputs and weights of the model to `int8`. This is done by `client.py` before sending the inputs and weights to the server. In the real-world use case, the weights would be quantised on the server before runtime, this is fine as the quantisation scale does not have the be the same for the weights and inputs.
The quantisation here is the source of the (negligible) difference in accuracy between the plaintext and encrypted models. Additional information about quantising ML models for use on our architecture can be found [here]()(WIP).

### Setup
- Requires download of [this](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download?datasetVersionNumber3) dataset.
	- Save as `creditcard.csv` in scripts folder.
- Requires installed Explore SDK (download [here](https://web.app.explore.optalysys.com/downloads.html), instructions for installation [here](https://web.app.explore.optalysys.com/getting_started/installation.html))

### Running locally

In order to see the demo in action:
- Run client.py
    - Splits whole dataset into training data and testing data
    - Trains machine learning model in the clear on training data
    - Runs inference on testing data in clear to provide accuracy baseline
    - Encrypts testing dataset ready for use in later steps
- Run server.py
    - Reads in encrypted testing dataset
    - Runs logistic classification on encrypted testing dataset
    - Returns encrypted classification results
- Run client\_decrypt.py
    - Reads in encrypted results
    - Decrypts results
    - Displays result of homomorphic evaluation against real classification of each transaction

```
cd scripts 

python3 client.py 1

python3 circuit.py local_fraud_demo

python3 client_decrypt local_fraud_demo
```

### Running on Explore

```
cd scripts

python3 client.py 1

explore run --name explore_fraud_demo spec circuit.py ../input/*

python3 client_decrypt explore_fraud_demo
```
