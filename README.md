#  TP1: Introduction to Flower Framework – Federated Learning Project

This repository contains my implementation of a Federated Learning (FL) system using the framework as part of the TP1 assignment. The project explores client-server FL simulation with the Fashion MNIST dataset, model training on distributed data, and custom FL strategy implementation.

##  Project Overview

In this TP, I explored the complete pipeline of building a federated learning system from scratch using Flower. Below are the key steps and what I accomplished in each:

###  Step 1: Data Generation & Loading

- data_utils.py: Generates distributed FashionMNIST datasets using Dirichlet distribution (saved in client_data) and loads client-specific DataLoader objects.

###  Step 2: Model Implementation

model.py: Defines CustomFashionModel, a PyTorch CNN with methods for training, evaluation, and parameter management.

###  Step 3: Federated Client

client.py: Implements CustomClient (extends flwr.client.Client) for local training, evaluation, and parameter exchange.

###  Step 4: Running a Client

run_client.py: Runs a client with python run_client.py --cid INTEGER. Requires an active server.

###  Step 5: Server Components

server.py: Implements CustomClientManager for client management and FedAvgStrategy for FedAvg aggregation.

###  Step 6: Running the Server

start_server.py: Launches the server with python start_server.py. Saves results to fl_results.json.

###  Step 7: Result Analysis

analyze_results.py: Visualizes results with python analyze_results.py [fl_results.json]. Outputs loss/accuracy plots to figures and a table using prettytable.

###  Step 8: Running the Simulation

run_simulation.py: Orchestrates the simulation:

Run python start_server.py.
Run python run_client.py --cid INTEGER for each client (e.g., 0 to 9).
Run python analyze_results.py to visualize results.

### Hyperparameters
Initial: 10 clients, 30 rounds, 1 epoch, α=1.0, batch size=32, learning rate=0.01.

### Experiments:
Round 1: 10 clients, α=1.0, 30 rounds, 1 epoch. Loss: 1.38→0.38, Eval Accuracy: 66.5%→86.1%.
Plots: Loss, Accuracy

Round 2: α=0.5 (more heterogeneity).
Round 3: 3 epochs.
Round 4: 5 clients, 3 epochs.

### Analysis
Lower α increases data heterogeneity, potentially slowing convergence.
More epochs improve local training but risk overfitting.
Fewer clients may reduce stability but speed up simulation.

This project demonstrates a complete FL system with data distribution, client-server interaction, and hyperparameter analysis.

