#  TP1: Introduction to Flower Framework – Federated Learning Project

This repository contains my implementation of a Federated Learning (FL) system using the framework as part of the TP1 assignment. The project explores client-server FL simulation with the Fashion MNIST dataset, model training on distributed data, and custom FL strategy implementation.

-

##  Project Overview

In this TP, I explored the complete pipeline of building a federated learning system from scratch using Flower. Below are the key steps and what I accomplished in each:

###  Step 1: Generate Distributed Dataset
- Created a simulated distributed dataset using the Dirichlet distribution.
- Ensured varying class distributions across clients to simulate data heterogeneity.

###  Step 2: Model Design
- Designed and implemented a custom PyTorch model for image classification.
- Included methods for training, evaluation, and parameter management.

###  Step 3: Federated Client Implementation
- Extended `flwr.client.Client` to define local training, evaluation, and parameter exchange logic.

###  Step 4: Run Individual Clients
- Developed a script (`run_client.py`) that launches a single client with its dataset and model.

###  Step 5: Server’s Client Manager
- Implemented a custom `ClientManager` to handle client registration, sampling, and management.

###  Step 6: Federated Strategy (FedAvg)
- Implemented the FedAvg aggregation logic by extending `flwr.server.Strategy`.

###  Step 7: Run the FL Server
- Created a server script (`run_server.py`) to start the Flower server with the custom client manager and strategy.

###  Step 8: Full FL Simulation
- Developed a script (`run_simulation.py`) to execute the full federated learning simulation with multiple clients.

###  Step 9: Result Analysis
- Ran experiments with different hyperparameters (clients, rounds, α value, etc.).
- Visualized and analyzed model convergence and accuracy using a custom `ResultsVisualizer`.


