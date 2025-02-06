import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import networkx as nx
import time
import tempfile
import random
from datetime import datetime, timedelta
import plotly.graph_objects as go
from pyvis.network import Network
import streamlit.components.v1 as components
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from shapely.geometry import Point, Polygon

st.set_page_config(page_title="Advanced Network Digital Twin", layout="wide")
st.markdown("""
<style>
/* Set overall background and typography */
body {
    background-color: #f4f4f4;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}

/* Main container styling */
.main {
    padding: 1rem;
}

/* Header styling */
h1, h2, h3, h4 {
    color: #333333;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}

/* Page title styling */
.stApp [data-testid="stHeader"] {
    background-color: #ffffff;
    border-bottom: 1px solid #dddddd;
}

/* Sidebar styling */
[data-testid="stSidebar"] .css-1d391kg {
    background-color: #ffffff;
    padding: 1rem;
}

/* Button styling */
.stButton > button {
    background-color: #4a90e2;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1rem;
}

/* Section header divider */
.section-header {
    border-bottom: 2px solid #cccccc;
    margin-top: 1rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
}

/* Customize tab headers */
div[data-baseweb="tab-list"] {
    background-color: #ffffff;
    border-bottom: 1px solid #dddddd;
}

/* DataFrame styling for better readability */
.css-1oe6wy0 {
    border: 1px solid #dddddd;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

class NetworkDigitalTwin:
    """Simulates a cellular network with dynamic topology and telemetry.
       Supports manual editing of the network.
    """
    def __init__(self, num_nodes):
        self.topology = self._create_hexagonal_topology(num_nodes)
        self.time = datetime.now()
        self.node_states = {i: {'base_connections': 50} for i in self.topology.nodes()}

    def _create_hexagonal_topology(self, num_nodes):
        """Generate a hexagonal grid topology with nodes placed only on land.
           A Shapely polygon (defined here as a simple rectangle) approximates an inland
           area of Abu Dhabi (avoiding water surfaces). Only points inside this polygon are accepted.
        """
        # Define a simple land polygon (longitude, latitude)
        land_polygon = Polygon([
            (54.34, 24.42),
            (54.34, 24.48),
            (54.40, 24.48),
            (54.40, 24.42)
        ])
        
        G = nx.Graph()
        grid_size = int(np.sqrt(num_nodes))
        
        min_lon, max_lon = 54.34, 54.40
        min_lat, max_lat = 24.42, 24.48
        
        candidate_points = []
        if grid_size > 1:
            delta_lon = (max_lon - min_lon) / (grid_size - 1)
            delta_lat = (max_lat - min_lat) / (grid_size - 0.5)
        else:
            delta_lon, delta_lat = 0, 0
        
        for i in range(grid_size * grid_size):
            row = i // grid_size
            col = i % grid_size
            lon = min_lon + col * delta_lon
            lat = min_lat + row * delta_lat + (col % 2) * (delta_lat / 2)
            if land_polygon.contains(Point(lon, lat)):
                candidate_points.append((lon, lat))
        
        while len(candidate_points) < num_nodes:
            rand_lon = random.uniform(min_lon, max_lon)
            rand_lat = random.uniform(min_lat, max_lat)
            if land_polygon.contains(Point(rand_lon, rand_lat)):
                candidate_points.append((rand_lon, rand_lat))
        
        candidate_points = candidate_points[:num_nodes]
        for idx, pos in enumerate(candidate_points):
            G.add_node(idx, type="base_station", pos=pos)
        
        for i in range(len(candidate_points)):
            if (i + 1) < len(candidate_points) and (i % grid_size) != (grid_size - 1):
                G.add_edge(i, i + 1)
            if (i + grid_size) < len(candidate_points):
                G.add_edge(i, i + grid_size)
                if (i % grid_size) != 0:
                    G.add_edge(i, i + grid_size - 1)
        return G

    def add_node(self, pos, capacity):
        """Add a new node with a given position and capacity."""
        new_id = int(max(self.topology.nodes()) + 1) if self.topology.nodes() else 0
        self.topology.add_node(new_id, type="user_node", pos=pos)
        self.node_states[new_id] = {'base_connections': capacity}
        return new_id

    def remove_node(self, node_id):
        """Remove a node and its edges."""
        if node_id in self.topology.nodes():
            self.topology.remove_node(node_id)
            if node_id in self.node_states:
                del self.node_states[node_id]

    def add_edge(self, node1, node2):
        """Add an edge between two nodes."""
        if node1 in self.topology.nodes() and node2 in self.topology.nodes():
            self.topology.add_edge(node1, node2)

    def remove_edge(self, node1, node2):
        """Remove the edge between two nodes."""
        if self.topology.has_edge(node1, node2):
            self.topology.remove_edge(node1, node2)

    def update_node(self, node_id, pos=None, capacity=None):
        """Update a node's position and/or capacity."""
        if node_id in self.topology.nodes():
            if pos is not None:
                self.topology.nodes[node_id]['pos'] = pos
            if capacity is not None:
                self.node_states[node_id]['base_connections'] = capacity

    def generate_telemetry(self):
        """Generate telemetry data for all nodes."""
        np.random.seed(int(self.time.timestamp()))
        base_throughput = 5 + np.sin(self.time.hour / 24 * 2 * np.pi) * 2
        user_peak = 100 + (self.time.hour % 12) * 20
        timestamp = np.datetime64(self.time)
        node_ids = list(self.topology.nodes())
        n = len(node_ids)
        data = pd.DataFrame({
            'timestamp': [timestamp] * n,
            'node_id': node_ids,
            'latency_ms': np.random.exponential(2, n),
            'throughput_gbps': np.random.normal(base_throughput, 1, n),
            'connected_users': np.random.poisson(user_peak, n),
            'packet_loss': np.random.uniform(0.1, 5, n),
            'signal_quality': np.random.normal(75, 10, n),
            'congestion_risk': np.zeros(n)
        })
        for idx, row in data.iterrows():
            node = row['node_id']
            capacity = self.node_states[node]['base_connections']
            data.at[idx, 'connected_users'] = min(row['connected_users'], capacity)
        data['congestion_risk'] = self._calculate_risk(data)
        self.time += timedelta(minutes=5)
        return data

    def _calculate_risk(self, data):
        """Calculate risk based on utilization and performance."""
        risks = []
        for i, row in data.iterrows():
            capacity = self.node_states[row['node_id']]['base_connections']
            utilization = row['connected_users'] / capacity if capacity > 0 else 1
            risk = (0.4 * utilization) + (0.2 * row['latency_ms']) + (0.2 * row['packet_loss']) + (0.2 * (100 - row['signal_quality']) / 100)
            risks.append(risk)
        return np.array(risks)

class CongestionPredictor(nn.Module):
    """LSTM-based model with attention to predict congestion risk."""
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
        if x.size(-1) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {x.size(-1)}")
        out, _ = self.lstm(x)
        attn_weights = self.attention(out)
        context = torch.sum(attn_weights * out, dim=1)
        return self.fc(context)

class NetworkOptimizer(gym.Env):
    """Gym environment for RL-based network optimization."""
    def __init__(self, network_twin):
        super().__init__()
        self.twin = network_twin
        self.num_nodes = len(self.twin.topology.nodes())
        self.action_space = gym.spaces.Discrete(self.num_nodes)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_nodes * 6,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        """Reset and return observation and empty info."""
        data = self.twin.generate_telemetry()
        obs = self._process_observation(data)
        return obs.astype(np.float32), {}
    
    def step(self, action):
        """
        Apply an action that boosts a node's capacity and simulates improved performance.
        """
        data = self.twin.generate_telemetry()
        node_idx = action % len(self.twin.topology.nodes())
        self.twin.node_states[node_idx]['base_connections'] = int(
            self.twin.node_states[node_idx]['base_connections'] * 1.2
        )
        idx = data.index[data['node_id'] == node_idx][0]
        data.at[idx, 'latency_ms'] *= 0.8      
        data.at[idx, 'packet_loss'] *= 0.8     
        data.at[idx, 'signal_quality'] = min(data.at[idx, 'signal_quality'] + 5, 100)
        cap = self.twin.node_states[node_idx]['base_connections']
        data.at[idx, 'connected_users'] = min(data.at[idx, 'connected_users'], cap)
        data['congestion_risk'] = self.twin._calculate_risk(data)
        obs = self._process_observation(data)
        reward = (100 - data['congestion_risk'].mean()) / 10
        done = False
        truncated = False
        info = {}
        return obs.astype(np.float32), reward, done, truncated, info
    
    def _process_observation(self, data):
        return data[['latency_ms', 'throughput_gbps', 'connected_users', 'packet_loss', 'signal_quality', 'congestion_risk']].values.flatten()

def calculate_risk_color(risk):
    """Return a color based on normalized congestion risk."""
    if risk > 0.9:
        return "#ff0000"
    return f"hsl({(1 - risk) * 120}, 70%, 50%)"

def render_network_dashboard(data, show_edges=True):
    """Render a 2D network topology using PyVis and automatically fit the view."""
    G = st.session_state.twin.topology.copy()
    node_data = data.set_index('node_id')
    for node in G.nodes():
        risk = float(node_data.loc[node, 'congestion_risk'])
        users = int(node_data.loc[node, 'connected_users'])
        latency = float(node_data.loc[node, 'latency_ms'])
        throughput = float(node_data.loc[node, 'throughput_gbps'])
        signal = float(node_data.loc[node, 'signal_quality'])
        G.nodes[node]['title'] = (
            f"Base Station {node}<br>"
            f"Signal Quality: {signal:.1f}%<br>"
            f"Active Connections: {users}<br>"
            f"Latency: {latency:.1f}ms<br>"
            f"Throughput: {throughput:.1f}Gbps<br>"
            f"Congestion Risk: {risk:.2f}"
        )
        G.nodes[node]['color'] = calculate_risk_color(risk)
        G.nodes[node]['size'] = 10 + risk * 20
    nt = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
    nt.from_nx(G)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        nt.save_graph(tmp.name)
        html_content = open(tmp.name, encoding="utf-8").read()

    html_content += """
    <script type="text/javascript">
      window.onload = function() {
          if (typeof network !== 'undefined') {
              network.fit();
          }
      };
    </script>
    """
    components.html(html_content, height=600, scrolling=True)

def render_3d_network_dashboard(data):
    """Render a 3D network topology visualization using Plotly with a zoomed-out camera view."""
    G = st.session_state.twin.topology.copy()
    node_data = data.set_index('node_id')
    node_x, node_y, node_z = [], [], []
    node_color = []
    node_text = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        risk = float(node_data.loc[node, 'congestion_risk'])
        node_x.append(x)
        node_y.append(y)
        node_z.append(risk * 0.5)
        node_color.append(risk)
        node_text.append(f"Base Station {node}<br>Risk: {risk:.2f}")
    scatter = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(size=8, color=node_color, colorscale='Viridis', colorbar=dict(title='Risk')),
        text=node_text,
        hoverinfo='text'
    )
    edge_traces = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        risk0 = float(node_data.loc[edge[0], 'congestion_risk'])
        risk1 = float(node_data.loc[edge[1], 'congestion_risk'])
        z0 = risk0 * 0.5
        z1 = risk1 * 0.5
        edge_trace = go.Scatter3d(
            x=[x0, x1, None],
            y=[y0, y1, None],
            z=[z0, z1, None],
            mode='lines',
            line=dict(color='gray', width=2),
            hoverinfo='none'
        )
        edge_traces.append(edge_trace)
    fig = go.Figure(data=[scatter] + edge_traces)
    fig.update_layout(
        title="3D Network Topology Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Risk (scaled)",
            camera=dict(eye=dict(x=3, y=3, z=3)) 
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    st.plotly_chart(fig, use_container_width=True)

def display_system_health(data):
    """Display key performance metrics of the network."""
    risk_mean = data['congestion_risk'].mean()
    total_connections = data['connected_users'].sum()
    avg_signal = data['signal_quality'].mean()
    high_risk = len(data[data['congestion_risk'] > 0.9])
    st.markdown("#### System Health Metrics")
    st.write(f"Total Connections: {total_connections}")
    st.write(f"Average Congestion Risk: {risk_mean:.2f}")
    st.write(f"Average Signal Quality: {avg_signal:.1f}%")
    st.write(f"High Risk Nodes (Risk > 0.9): {high_risk}")

def display_risk_predictions(data):
    """Display a radar chart of congestion risk across nodes."""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=data['congestion_risk'],
        theta=data['node_id'] * (360 / len(data)),
        mode='markers',
        marker=dict(
            size=data['connected_users'] / 5,
            color=data['congestion_risk'],
            colorscale='Portland',
            showscale=True,
            line=dict(width=2, color='DarkSlateGrey')
        ),
        hoverinfo='text',
        hovertext=[f"Node {row['node_id']}<br>Risk: {row['congestion_risk']:.2f}" for _, row in data.iterrows()]
    ))
    fig.update_layout(
        height=300,
        title="Network Risk Radar",
        template="plotly_white",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.5]),
            angularaxis=dict(showticklabels=False)
        ),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

def display_model_training_data(sequences, targets):
    """Display sample training data for the congestion predictor model."""
    st.markdown("#### Sample Training Data")
    st.write("Each sample is a sequence of telemetry features with a target congestion risk value.")
    sample_seq = sequences.numpy() if isinstance(sequences, torch.Tensor) else sequences
    sample_target = targets.numpy() if isinstance(targets, torch.Tensor) else targets
    st.dataframe(pd.DataFrame(sample_seq.reshape(-1, sample_seq.shape[-1])[:10],
                                columns=["connected_users", "latency_ms", "throughput_gbps", "packet_loss", "signal_quality", "congestion_risk"]))
    st.write("Target values:")
    st.write(sample_target[:10])

def inject_network_anomaly():
    """Inject anomalies by reducing capacity for a few random nodes."""
    twin = st.session_state.twin
    affected_nodes = random.sample(list(twin.topology.nodes()), 3)
    for node in affected_nodes:
        twin.node_states[node]['base_connections'] = int(twin.node_states[node]['base_connections'] * 0.5)
    st.success(f"Anomaly injected in nodes: {affected_nodes}")

def render_real_project_map():
    """
    Render an interactive map using st.map to display network nodes as real-world project locations.
    Each node's 'pos' tuple is interpreted as (longitude, latitude).
    """
    nodes_list = []
    for node_id, attrs in st.session_state.twin.topology.nodes(data=True):
        pos = attrs.get("pos", (None, None))
        if pos[0] is not None and pos[1] is not None:
            nodes_list.append({
                "lat": pos[1],
                "lon": pos[0],
                "node_id": node_id,
                "capacity": st.session_state.twin.node_states[node_id]['base_connections']
            })
    if nodes_list:
        df_nodes = pd.DataFrame(nodes_list)
        st.markdown("### Real Projects Map")
        st.map(df_nodes)
    else:
        st.warning("No nodes available for mapping.")

def prepare_training_data(data, seq_length=24):
    """Prepare training data from telemetry data (sequence length 24)."""
    features = data[['connected_users', 'latency_ms', 'throughput_gbps', 'packet_loss', 'signal_quality', 'congestion_risk']].values
    if len(features) < seq_length + 1:
        return None, None
    sequences = []
    targets = []
    num_samples = len(features) - seq_length
    for i in range(num_samples):
        sequences.append(features[i:i+seq_length])
        targets.append(features[i+seq_length][-1])
    sequences = torch.from_numpy(np.array(sequences, dtype=np.float32))
    targets = torch.from_numpy(np.array(targets, dtype=np.float32))
    return sequences, targets

def train_model(model, criterion, optimizer, sequences, targets):
    """Train the congestion predictor model for 100 epochs."""
    dataset = TensorDataset(sequences, targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model.train()
    loss_history = []
    progress_bar = st.progress(0)
    epochs = 100
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_seq, batch_target in loader:
            optimizer.zero_grad()
            outputs = model(batch_seq)
            loss = criterion(outputs.squeeze(), batch_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(loader))
        progress_bar.progress((epoch + 1) / epochs)
    return loss_history

def execute_training_workflow(data):
    """Run the training workflow for the congestion predictor model."""
    st.markdown("## Predictive Analytics Model Training")
    sequences, targets = prepare_training_data(data)
    if sequences is None:
        st.error("Insufficient telemetry data. Collect at least 24 samples for training.")
        return
    display_model_training_data(sequences, targets)
    st.write("Initializing and training the congestion predictor model...")
    model = CongestionPredictor(input_size=6)
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    loss_history = train_model(model, criterion, optimizer, sequences, targets)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_history, mode='lines+markers', name='Training Loss'))
    fig.update_layout(title='Model Training Loss over Epochs', xaxis_title='Epoch', yaxis_title='Loss')
    st.plotly_chart(fig, use_container_width=True)
    st.success("Model training completed.")
    st.session_state.model = model

def execute_optimization_workflow(data):
    """Run the RL-based network optimization workflow."""
    st.markdown("## Autonomous Optimization")
    st.write("The RL agent selects a node to optimize based on current network metrics.")
    env = make_vec_env(lambda: NetworkOptimizer(st.session_state.twin), n_envs=1)
    with st.spinner("Training the RL agent..."):
        rl_model = PPO('MlpPolicy', env, verbose=0)
        rl_model.learn(total_timesteps=1000)
    st.success("RL agent training complete.")
    st.write("Applying the learned optimization action...")
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    prediction = rl_model.predict(obs)
    action = int(prediction[0][0]) if isinstance(prediction, tuple) and isinstance(prediction[0], (list, np.ndarray)) else int(prediction)
    st.write(f"The RL agent has selected node {action} for optimization.")
    data_before = st.session_state.twin.generate_telemetry()
    before_data = data_before.copy()
    _ = env.step(np.array([action]))
    after_data = data_before.copy()
    node_idx = action % len(st.session_state.twin.topology.nodes())
    idx = after_data.index[after_data['node_id'] == node_idx][0]
    after_data.at[idx, 'latency_ms'] *= 0.8
    after_data.at[idx, 'packet_loss'] *= 0.8
    after_data.at[idx, 'signal_quality'] = min(after_data.at[idx, 'signal_quality'] + 5, 100)
    capacity = st.session_state.twin.node_states[node_idx]['base_connections']
    after_data.at[idx, 'connected_users'] = min(after_data.at[idx, 'connected_users'], capacity)
    after_data['congestion_risk'] = st.session_state.twin._calculate_risk(after_data)
    st.subheader("Network State Before Optimization")
    render_network_dashboard(before_data)
    display_system_health(before_data)
    st.dataframe(before_data)
    st.subheader("Network State After Optimization")
    render_network_dashboard(after_data)
    display_system_health(after_data)
    st.dataframe(after_data)
    display_optimization_impact(action, before_data, after_data)

def display_optimization_impact(action, before_data, after_data):
    """Display the detailed impact of the optimization on a selected node."""
    node_idx = action % len(before_data)
    before_node = before_data[before_data['node_id'] == node_idx].iloc[0]
    after_node = after_data[after_data['node_id'] == node_idx].iloc[0]
    st.markdown("#### Optimization Impact on Selected Node")
    st.write(f"Node: {node_idx}")
    st.write("Before Optimization:")
    st.write(before_node)
    st.write("After Optimization:")
    st.write(after_node)
    risk_diff = before_node['congestion_risk'] - after_node['congestion_risk']
    st.write(f"Risk Improvement: {risk_diff:.2f} (positive indicates improvement)")

def simulate_real_network(iterations=5, delay=2):
    """Simulate a real network over several iterations."""
    st.markdown("## Real Network Simulation")
    for i in range(iterations):
        data = st.session_state.twin.generate_telemetry()
        st.markdown(f"**Iteration {i+1} Telemetry Data**")
        st.dataframe(data)
        render_network_dashboard(data)
        time.sleep(delay)
        st.markdown("---")
    st.success("Simulation complete.")

def main():
    st.title("Advanced Network Digital Twin and Optimization Platform")
    
    if 'twin' not in st.session_state:
        st.session_state.twin = NetworkDigitalTwin(num_nodes=49)
        st.session_state.model = None
    
    st.sidebar.header("Configuration")
    simulation_speed = st.sidebar.slider("Simulation Speed (seconds delay)", 1, 10, 3)
    viz_mode = st.sidebar.radio("Visualization Mode", options=["2D", "3D"])
    if st.sidebar.button("Inject Network Anomaly"):
        inject_network_anomaly()
        st.experimental_rerun()
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Dashboard", "Training", "Optimization", "Simulation", "Create Your Own Network Playground", "About"
    ])
    
    with tab1:
        st.header("Dashboard")
        current_data = st.session_state.twin.generate_telemetry()
        st.subheader("Current Telemetry Data")
        st.dataframe(current_data)
        if viz_mode == "2D":
            render_network_dashboard(current_data)
        else:
            render_3d_network_dashboard(current_data)
        display_system_health(current_data)
        display_risk_predictions(current_data)
    
    with tab2:
        st.header("Predictive Analytics")
        current_data = st.session_state.twin.generate_telemetry()
        execute_training_workflow(current_data)
    
    with tab3:
        st.header("Autonomous Optimization")
        current_data = st.session_state.twin.generate_telemetry()
        execute_optimization_workflow(current_data)
    
    with tab4:
        st.header("Real Network Simulation")
        simulate_real_network(iterations=5, delay=simulation_speed)
    
    with tab5:
        st.header("Create Your Own Network Playground")
        st.write("Customize your network by adding, removing, connecting, and modifying nodes (cell towers).")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Add a Node")
            new_lat = st.number_input("Latitude", value=24.45, step=0.001, key="add_lat")
            new_lon = st.number_input("Longitude", value=54.37, step=0.001, key="add_lon")
            new_capacity = st.number_input("Node Capacity", value=50, step=1, key="add_cap")
            if st.button("Add Node"):
                new_id = st.session_state.twin.add_node((new_lon, new_lat), int(new_capacity))
                st.success(f"Node {new_id} added at (lat: {new_lat}, lon: {new_lon}) with capacity {new_capacity}.")
                st.experimental_rerun()
        with col2:
            st.subheader("Remove a Node")
            rem_node = st.number_input("Node ID to Remove", value=0, step=1, key="rem_node")
            if st.button("Remove Node"):
                if rem_node in st.session_state.twin.topology.nodes():
                    st.session_state.twin.remove_node(int(rem_node))
                    st.success(f"Node {rem_node} removed.")
                    st.experimental_rerun()
                else:
                    st.error(f"Node {rem_node} does not exist.")
        st.subheader("Connect Nodes")
        col3, col4 = st.columns(2)
        with col3:
            node_a = st.selectbox("Select Node A", list(st.session_state.twin.topology.nodes()), key="node_a")
        with col4:
            node_b = st.selectbox("Select Node B", list(st.session_state.twin.topology.nodes()), key="node_b")
        if st.button("Connect Nodes"):
            if node_a != node_b:
                st.session_state.twin.add_edge(node_a, node_b)
                st.success(f"Connected Node {node_a} and Node {node_b}.")
                st.experimental_rerun()
            else:
                st.error("Cannot connect a node to itself.")
        st.subheader("Remove an Edge")
        col5, col6 = st.columns(2)
        with col5:
            edge_a = st.selectbox("Select Node A for Edge", list(st.session_state.twin.topology.nodes()), key="edge_a")
        with col6:
            edge_b = st.selectbox("Select Node B for Edge", list(st.session_state.twin.topology.nodes()), key="edge_b")
        if st.button("Remove Edge"):
            if st.session_state.twin.topology.has_edge(edge_a, edge_b):
                st.session_state.twin.remove_edge(edge_a, edge_b)
                st.success(f"Edge between Node {edge_a} and Node {edge_b} removed.")
                st.experimental_rerun()
            else:
                st.error("The specified edge does not exist.")
        st.subheader("Modify a Node")
        col7, col8 = st.columns(2)
        with col7:
            mod_node = st.selectbox("Select Node to Modify", list(st.session_state.twin.topology.nodes()), key="mod_node")
        with col8:
            new_pos_lat = st.number_input("New Latitude", value=24.45, step=0.001, key="mod_lat")
            new_pos_lon = st.number_input("New Longitude", value=54.37, step=0.001, key="mod_lon")
            new_cap = st.number_input("New Capacity", value=50, step=1, key="mod_cap")
        if st.button("Update Node"):
            st.session_state.twin.update_node(mod_node, pos=(new_pos_lon, new_pos_lat), capacity=int(new_cap))
            st.success(f"Node {mod_node} updated.")
            st.experimental_rerun()
        st.subheader("Current Network Configuration")
        nodes_data = []
        for n in st.session_state.twin.topology.nodes(data=True):
            node_id = n[0]
            pos = n[1].get("pos", (None, None))
            capacity = st.session_state.twin.node_states.get(node_id, {}).get("base_connections", None)
            nodes_data.append({
                "node_id": np.float64(int(node_id)),
                "lon": np.float64(float(pos[0])) if pos[0] is not None else None,
                "lat": np.float64(float(pos[1])) if pos[1] is not None else None,
                "capacity": np.float64(int(capacity)) if capacity is not None else None
            })
        st.dataframe(pd.DataFrame(nodes_data))
        st.write("After editing your network, visit the Dashboard, Training, or Optimization tabs to see how your network performs.")
        st.markdown("---")
        if st.button("Show Real Projects Map"):
            render_real_project_map()
    
    with tab6:
        st.header("About This Platform")
        st.markdown("""
**Advanced Network Digital Twin and Optimization Platform**

This platform simulates a dynamic cellular network using a digital twin approach. Features include:

- Dynamic Network Topology: Visualize and customize the network in both 2D and 3D.
- Real-Time Telemetry: Monitor metrics such as latency, throughput, connected users, packet loss, signal quality, and congestion risk.
- Predictive Analytics: Train an LSTM-based congestion predictor with attention mechanisms.
- Reinforcement Learning: Optimize network parameters autonomously using a PPO agent.
- Continuous Simulation: Observe network evolution over time.
- Create Your Own Network Playground: Add, remove, modify, and connect nodes interactively.
- Interactive Real Map: View network nodes as points on a real-world map to demonstrate real projects.
- Anomaly Injection: Simulate network disruptions to test resilience.

Developed using Streamlit, PyVis, Plotly, Stable Baselines3, PyTorch, and Shapely.
""")
    
if __name__ == "__main__":
    main()
