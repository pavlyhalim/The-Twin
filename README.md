# Network Cognitive Manager Simulator

# Live demo 

# https://the-twin.streamlit.app/

## Overview

The **Network Cognitive Manager Simulator** is an AI-driven network optimization system that simulates a dynamic network environment using randomly generated data. This project demonstrates how to generate network telemetry, visualize network topology, predict network congestion, and perform reinforcement learning–based optimization—all in one integrated simulator.


## Features

- **Dynamic Network Simulation**:  
  Generates random telemetry data including latency, throughput, connected users, packet loss, signal quality, and calculated congestion risk.

- **Interactive Visualization**:  
  Visualizes the network topology using Pyvis and displays performance metrics with Plotly charts and Streamlit components.

- **AI/ML Integration**:  
  Incorporates an LSTM-based neural network with attention for predicting congestion risk from historical telemetry data.

- **Reinforcement Learning Optimization**:  
  Implements a reinforcement learning environment using PPO (Proximal Policy Optimization) from Stable Baselines3 to optimize network configurations.

- **Web Interface**:  
  Provides a user-friendly dashboard built with Streamlit to interact with the simulator in real time.

## Installation

Ensure you have Python 3.12 or later installed. Then, install the required dependencies using pip:

```bash
pip install streamlit torch numpy pandas plotly networkx pyvis stable-baselines3 gymnasium
```

## Usage

Run the simulator with Streamlit:
```bash
streamlit run app.py
```


## Random Data Simulation

This simulator generates random data for demonstration purposes. Metrics such as latency, throughput, and signal quality are produced using random distributions, allowing you to test and visualize network behavior in a controlled, simulated environment.
