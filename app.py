import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium import spaces

class NetworkDigitalTwin:
    """Virtual network simulator with dynamic topology"""
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.topology = self._create_topology()
        self.time = datetime.now()
        
    def _create_topology(self):
        """Generate hexagonal cellular network pattern"""
        G = nx.Graph()
        cluster_size = 7
        for i in range(self.num_nodes):
            G.add_node(i, type="base_station", 
                      pos=(np.random.rand(), np.random.rand()))
            if i % cluster_size != 0:
                G.add_edge(i, i-1)
        return G
    
    def generate_telemetry(self):
        """Generate simulated network metrics with temporal patterns"""
        np.random.seed(int(self.time.timestamp()))
        
        base_throughput = 5 + np.sin(self.time.hour/24*2*np.pi)*2
        user_peak = 100 + (self.time.hour % 12)*20
        
        data = pd.DataFrame({
            'timestamp': self.time,
            'node_id': range(self.num_nodes),
            'latency_ms': np.random.exponential(2, self.num_nodes),
            'throughput_gbps': np.random.normal(base_throughput, 1, self.num_nodes),
            'connected_users': np.random.poisson(user_peak, self.num_nodes),
            'packet_loss': np.random.uniform(0.1, 5, self.num_nodes),
            'signal_quality': np.random.normal(75, 10, self.num_nodes),
            'congestion_risk': np.zeros(self.num_nodes)
        })
        
        data['congestion_risk'] = self._calculate_risk(data)
        self.time += timedelta(minutes=5)
        return data
    
    def _calculate_risk(self, data):
        """Dynamic risk calculation with time-based weighting"""
        return (data['connected_users'] * 0.4 +
                data['latency_ms'] * 0.2 +
                data['packet_loss'] * 0.2 +
                (100 - data['signal_quality']) * 0.2)

class CongestionPredictor(nn.Module):
    """Enhanced temporal predictor with input validation"""
    def __init__(self, input_size=6, hidden_size=64):
        super().__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Validate input dimensions
        if x.size(-1) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {x.size(-1)}")
            
        out, _ = self.lstm(x)
        attn_weights = self.attention(out)
        context = torch.sum(attn_weights * out, dim=1)
        return self.fc(context)

class NetworkOptimizer(gym.Env):
    """Reinforcement learning environment for traffic management"""
    def __init__(self, network_twin):
        super().__init__()
        self.twin = network_twin
        self.num_nodes = self.twin.num_nodes
        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_nodes * 6,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        data = self.twin.generate_telemetry()
        obs = self._process_observation(data)
        return obs.astype(np.float32), {}

    def step(self, action):
        current_data = self.twin.generate_telemetry()
        node_idx = action % self.num_nodes
        
        # Apply mitigation strategy
        current_data.loc[node_idx, 'connected_users'] *= 0.8
        current_data.loc[node_idx, 'throughput_gbps'] *= 1.2
        
        obs = self._process_observation(current_data)
        reward = (100 - current_data['congestion_risk'].mean()) / 10
        return obs.astype(np.float32), reward, False, False, {}

    def _process_observation(self, data):
        """Feature engineering for RL observations"""
        return data[['latency_ms', 'throughput_gbps', 
                    'connected_users', 'packet_loss',
                    'signal_quality', 'congestion_risk']].values.flatten()


def main():
    st.set_page_config(page_title="Network Cognitive Manager", layout="wide")
    st.title("The TWin - Digital Twin-Based Network Optimization")
    
    # System initialization
    if 'twin' not in st.session_state:
        st.session_state.twin = NetworkDigitalTwin(num_nodes=49)
        st.session_state.training_history = []
    
    # Control panel
    with st.sidebar:
        st.header("System Configuration")
        st.button("Initialize New Network", help="Create fresh network instance")
        
        st.subheader("Analytics Controls")
        train_model = st.checkbox("Enable Predictive Analytics")
        optimize_network = st.checkbox("Enable Autonomous Optimization")
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Network Topology Visualization")
        current_state = st.session_state.twin.generate_telemetry()
        render_network_dashboard(current_state)
        
        st.subheader("Performance Metrics")
        display_system_health(current_state)
        
        st.subheader("Risk Forecasting")
        display_risk_predictions(current_state)

    with col2:
        st.header("Cognitive Operations")
        if train_model:
            execute_training_workflow(current_state)
                
        if optimize_network:
            execute_optimization_workflow(current_state)

def render_network_dashboard(data):
    """Interactive 3D network state visualization"""
    G = nx.Graph()
    for _, node in data.iterrows():
        node_id = str(int(node['node_id']))
        tooltip = f"""
        Base Station {node_id}
        Signal Quality: {node['signal_quality']:.1f}%
        Active Connections: {int(node['connected_users'])}
        Latency: {node['latency_ms']:.1f}ms
        Throughput: {node['throughput_gbps']:.1f}Gbps
        Risk Level: {node['congestion_risk']:.1f}%
        """
        G.add_node(node_id, title=tooltip, color=calculate_risk_color(node['congestion_risk']))
    
    nt = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
    nt.from_nx(G)
    nt.save_graph('nt.html')
    components.html(open('nt.html').read(), height=600)

def calculate_risk_color(risk):
    """Color gradient from green to red based on risk level"""
    return f"hsl({(100 - risk) * 1.2}, 70%, 50%)"

def display_system_health(data):
    """Real-time performance indicators"""
    metrics = st.columns(4)
    with metrics[0]:
        st.metric("Total Connections", f"{data['connected_users'].sum():,}")
    with metrics[1]:
        st.metric("Network Utilization", f"{data['congestion_risk'].mean():.1f}%")
    with metrics[2]:
        st.metric("Quality Degradation", f"{data['signal_quality'].mean():.1f}%")
    with metrics[3]:
        st.metric("High Risk Nodes", len(data[data['congestion_risk'] > 75]))

def execute_training_workflow(data):
    """End-to-end model training process"""
    with st.status("Training Predictive Model...", expanded=True) as status:
        st.write("Phase 1: Data Preparation")
        sequences, targets = prepare_training_data(data)
        
        st.write("Phase 2: Model Initialization")
        model = CongestionPredictor(input_size=6)
        criterion = nn.HuberLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        
        st.write("Phase 3: Model Training")
        train_model(model, criterion, optimizer, sequences, targets)
        
        status.update(label="Training Complete", state="complete")
    st.session_state.model = model
    st.success("Predictive model updated with latest network data")

def prepare_training_data(data, seq_length=12):
    """Convert temporal data to training sequences with manual windowing"""
    features = data[['connected_users', 'latency_ms', 
                    'throughput_gbps', 'packet_loss',
                    'signal_quality', 'congestion_risk']].values
    
    num_samples = len(features) - seq_length
    sequences = []
    targets = []
    
    for i in range(num_samples):
        sequences.append(features[i:i+seq_length])
        targets.append(features[i+seq_length][-1])  # Predict next congestion risk
    
    sequences_np = np.array(sequences, dtype=np.float32)
    targets_np = np.array(targets, dtype=np.float32)
    
    return (torch.from_numpy(sequences_np), 
            torch.from_numpy(targets_np))

def train_model(model, criterion, optimizer, sequences, targets):
    """Model training execution"""
    dataset = TensorDataset(sequences, targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(50):
        for batch_seq, batch_target in loader:
            optimizer.zero_grad()
            outputs = model(batch_seq)
            loss = criterion(outputs.squeeze(), batch_target)
            loss.backward()
            optimizer.step()

def execute_optimization_workflow(data):
    """Autonomous network optimization process"""
    with st.status("Executing Network Optimization...", expanded=True) as status:
        st.write("Phase 1: Environment Initialization")
        env = make_vec_env(lambda: NetworkOptimizer(st.session_state.twin), n_envs=1)
        
        st.write("Phase 2: Policy Learning")
        model = PPO('MlpPolicy', env, verbose=0)
        model.learn(total_timesteps=1000)
        
        st.write("Phase 3: Action Implementation")
        obs = env.reset()
        action, _ = model.predict(obs)
        # Convert action to a scalar integer
        action = int(action[0]) if isinstance(action, (list, np.ndarray)) else int(action)
        
        status.update(label="Optimization Complete", state="complete")
    display_optimization_impact(action, data)

def display_risk_predictions(data):
    """Interactive risk projection visualization"""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=data['node_id'],
        y=data['congestion_risk'],
        mode='markers',
        marker=dict(
            size=data['connected_users']/10,
            color=data['congestion_risk'],
            colorscale='Portland',
            showscale=True
        )
    ))
    fig.update_layout(
        height=300,
        title="Network Node Risk Distribution",
        xaxis_title="Node ID",
        yaxis_title="Risk Level (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

def display_optimization_impact(action, data):
    """Optimization results visualization"""
    if isinstance(action, (np.ndarray, list)):
        action = int(action[0])
    node_idx = action % len(data)
    node_data = data.iloc[node_idx]
    
    with st.expander("Optimization Details", expanded=True):
        st.subheader(f"Target Node {node_idx}")
        cols = st.columns(2)
        with cols[0]:
            st.metric("Current Risk Level", f"{node_data['congestion_risk']:.1f}%")
            st.metric("Connected Users", node_data['connected_users'])
        with cols[1]:
            st.metric("Throughput Capacity", f"{node_data['throughput_gbps']:.1f} Gbps")
            st.metric("Signal Quality", f"{node_data['signal_quality']:.1f}%")
        
        st.progress(0.3, text="Implementing traffic redistribution...")
        st.progress(0.6, text="Adjusting bandwidth allocation...")
        st.progress(0.9, text="Finalizing configuration changes...")


if __name__ == "__main__":
    main()
