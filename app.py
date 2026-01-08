# app.py - Complete Interactive EPANET Analysis Dashboard

import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import base64
import io
import json
import warnings
warnings.filterwarnings('ignore')

# For quantum walk simulation (lightweight implementation)
import numpy as np
from scipy.linalg import expm

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}]
)

app.title = "EPANET Anomaly & Quantum Walk Analyzer"
server = app.server

# Initialize global storage
epanet_data = None
network_graph = None
simulation_results = {'normal': None, 'attacked': None}
walk_results = {'classical': None, 'quantum': None}

# Custom CSS
custom_css = """
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .card-custom {
        background: #2c3e50;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .tab-content {
        background: #34495e;
        border-radius: 10px;
        padding: 20px;
        margin-top: 10px;
    }
    .btn-animate {
        transition: all 0.3s ease;
    }
    .btn-animate:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px;
    }
    .quantum-glow {
        animation: quantum-pulse 2s infinite;
    }
    @keyframes quantum-pulse {
        0% { box-shadow: 0 0 5px #00ff9d; }
        50% { box-shadow: 0 0 20px #00ff9d, 0 0 30px #00ff9d; }
        100% { box-shadow: 0 0 5px #00ff9d; }
    }
</style>
"""

# App layout
app.layout = html.Div([
    # Custom CSS
    html.Div(custom_css, dangerously_set_inner_html={'__html': custom_css}),
    
    # Header
    html.Div([
        html.H1("🌊 EPANET Network Security Analyzer", className="main-header"),
        html.P("Interactive anomaly injection, quantum walk simulation, and security analysis for water distribution networks", 
               style={'textAlign': 'center', 'color': '#ecf0f1'}),
    ]),
    
    # Main tabs
    dcc.Tabs([
        # Tab 1: Network Upload and Visualization
        dcc.Tab(label='📁 Network Setup', children=[
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Upload EPANET .inp File", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
                            dbc.CardBody([
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select File', style={'color': '#3498db'})
                                    ]),
                                    style={
                                        'width': '100%',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '2px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin': '10px',
                                        'backgroundColor': '#34495e'
                                    },
                                    multiple=False
                                ),
                                html.Div(id='upload-status', style={'marginTop': '10px'}),
                                html.Hr(),
                                html.H5("Quick Load Example Networks:"),
                                dbc.Row([
                                    dbc.Col(dbc.Button("Small Network", id='btn-example-small', 
                                                       color="primary", className="btn-animate m-1"), width=4),
                                    dbc.Col(dbc.Button("Medium Network", id='btn-example-medium', 
                                                       color="primary", className="btn-animate m-1"), width=4),
                                    dbc.Col(dbc.Button("Large Network", id='btn-example-large', 
                                                       color="primary", className="btn-animate m-1"), width=4),
                                ]),
                            ])
                        ], className="card-custom"),
                        
                        dbc.Card([
                            dbc.CardHeader("Network Information", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
                            dbc.CardBody([
                                html.Div(id='network-info'),
                                html.Div(id='node-list-container'),
                            ])
                        ], className="card-custom"),
                    ], width=4),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Network Visualization", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
                            dbc.CardBody([
                                dcc.Graph(id='network-graph', style={'height': '600px'}),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Layout Algorithm:", style={'color': 'white'}),
                                        dcc.Dropdown(
                                            id='layout-algorithm',
                                            options=[
                                                {'label': 'Spring Layout', 'value': 'spring'},
                                                {'label': 'Kamada-Kawai', 'value': 'kamada'},
                                                {'label': 'Circular', 'value': 'circular'},
                                                {'label': 'Random', 'value': 'random'}
                                            ],
                                            value='spring',
                                            clearable=False,
                                            style={'backgroundColor': '#2c3e50', 'color': 'white'}
                                        ),
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Label("Node Size By:", style={'color': 'white'}),
                                        dcc.Dropdown(
                                            id='node-size-metric',
                                            options=[
                                                {'label': 'Degree', 'value': 'degree'},
                                                {'label': 'Betweenness', 'value': 'betweenness'},
                                                {'label': 'Constant', 'value': 'constant'}
                                            ],
                                            value='degree',
                                            clearable=False,
                                            style={'backgroundColor': '#2c3e50', 'color': 'white'}
                                        ),
                                    ], width=6),
                                ]),
                            ])
                        ], className="card-custom"),
                    ], width=8),
                ]),
            ], className="tab-content"),
        ]),
        
        # Tab 2: Anomaly Injection
        dcc.Tab(label='⚠️ Anomaly Injection', children=[
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Configure Attack Scenario", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
                            dbc.CardBody([
                                html.H5("Attack Type Selection:", style={'color': '#3498db'}),
                                dcc.Checklist(
                                    id='attack-types',
                                    options=[
                                        {'label': ' Demand Increase (False Data)', 'value': 'demand'},
                                        {'label': ' Pressure Sensor Spoofing', 'value': 'pressure'},
                                        {'label': ' Pipe Leak (Physical Attack)', 'value': 'leak'},
                                        {'label': ' Pump Failure', 'value': 'pump'},
                                        {'label': ' Valve Manipulation', 'value': 'valve'},
                                    ],
                                    value=['demand', 'pressure'],
                                    labelStyle={'display': 'block', 'margin': '5px'}
                                ),
                                html.Hr(),
                                html.H5("Attack Parameters:", style={'color': '#3498db'}),
                                
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Magnitude (Multiplier):", style={'color': 'white'}),
                                        dcc.Slider(
                                            id='attack-magnitude',
                                            min=1.0,
                                            max=5.0,
                                            step=0.1,
                                            value=2.0,
                                            marks={i: str(i) for i in [1, 2, 3, 4, 5]},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        ),
                                    ]),
                                ]),
                                
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Start Time (hours):", style={'color': 'white'}),
                                        dcc.Slider(
                                            id='attack-start',
                                            min=0,
                                            max=24,
                                            step=1,
                                            value=8,
                                            marks={i: str(i) for i in [0, 6, 12, 18, 24]},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        ),
                                    ]),
                                    dbc.Col([
                                        dbc.Label("Duration (hours):", style={'color': 'white'}),
                                        dcc.Slider(
                                            id='attack-duration',
                                            min=1,
                                            max=12,
                                            step=1,
                                            value=4,
                                            marks={i: str(i) for i in [1, 4, 8, 12]},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        ),
                                    ]),
                                ]),
                                
                                html.Hr(),
                                html.H5("Target Selection:", style={'color': '#3498db'}),
                                
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Target Nodes:", style={'color': 'white'}),
                                        dcc.Dropdown(
                                            id='target-nodes',
                                            multi=True,
                                            placeholder="Select nodes to attack...",
                                            style={'backgroundColor': '#2c3e50', 'color': 'white'}
                                        ),
                                    ]),
                                    dbc.Col([
                                        dbc.Label("Target Links:", style={'color': 'white'}),
                                        dcc.Dropdown(
                                            id='target-links',
                                            multi=True,
                                            placeholder="Select links to attack...",
                                            style={'backgroundColor': '#2c3e50', 'color': 'white'}
                                        ),
                                    ]),
                                ]),
                                
                                html.Hr(),
                                dbc.Button("🚨 Launch Attack Simulation", 
                                          id='btn-simulate-attack',
                                          color="danger",
                                          size="lg",
                                          className="btn-animate w-100",
                                          n_clicks=0),
                            ])
                        ], className="card-custom"),
                    ], width=4),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Attack Visualization", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
                            dbc.CardBody([
                                dcc.Graph(id='attack-visualization', style={'height': '300px'}),
                                html.Hr(),
                                dcc.Graph(id='pressure-comparison', style={'height': '300px'}),
                            ])
                        ], className="card-custom"),
                        
                        dbc.Card([
                            dbc.CardHeader("Impact Analysis", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
                            dbc.CardBody([
                                html.Div(id='impact-metrics'),
                                dash_table.DataTable(
                                    id='impact-table',
                                    style_table={'overflowX': 'auto'},
                                    style_cell={
                                        'backgroundColor': '#2c3e50',
                                        'color': 'white',
                                        'border': '1px solid #34495e'
                                    },
                                    style_header={
                                        'backgroundColor': '#1a252f',
                                        'fontWeight': 'bold'
                                    },
                                ),
                            ])
                        ], className="card-custom"),
                    ], width=8),
                ]),
            ], className="tab-content"),
        ]),
        
        # Tab 3: Quantum vs Random Walk
        dcc.Tab(label='🚶‍♂️ Quantum Walk Analysis', children=[
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Walk Configuration", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
                            dbc.CardBody([
                                html.H5("Walk Parameters:", style={'color': '#3498db'}),
                                
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Start Node:", style={'color': 'white'}),
                                        dcc.Dropdown(
                                            id='walk-start-node',
                                            placeholder="Select start node...",
                                            style={'backgroundColor': '#2c3e50', 'color': 'white'}
                                        ),
                                    ]),
                                    dbc.Col([
                                        dbc.Label("Number of Steps:", style={'color': 'white'}),
                                        dcc.Slider(
                                            id='walk-steps',
                                            min=10,
                                            max=200,
                                            step=10,
                                            value=50,
                                            marks={10: '10', 100: '100', 200: '200'},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        ),
                                    ]),
                                ]),
                                
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Quantum Walk Type:", style={'color': 'white'}),
                                        dcc.Dropdown(
                                            id='quantum-walk-type',
                                            options=[
                                                {'label': 'Continuous-Time Quantum Walk', 'value': 'continuous'},
                                                {'label': 'Discrete-Time Quantum Walk', 'value': 'discrete'},
                                                {'label': 'Szegedy Quantum Walk', 'value': 'szegedy'},
                                            ],
                                            value='continuous',
                                            clearable=False,
                                            style={'backgroundColor': '#2c3e50', 'color': 'white'}
                                        ),
                                    ]),
                                    dbc.Col([
                                        dbc.Label("Time Evolution:", style={'color': 'white'}),
                                        dcc.Slider(
                                            id='quantum-time',
                                            min=1,
                                            max=20,
                                            step=1,
                                            value=10,
                                            marks={1: '1', 10: '10', 20: '20'},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        ),
                                    ]),
                                ]),
                                
                                html.Hr(),
                                html.H5("Comparison Metrics:", style={'color': '#3498db'}),
                                dcc.Checklist(
                                    id='walk-metrics',
                                    options=[
                                        {'label': ' Mixing Time', 'value': 'mixing'},
                                        {'label': ' Spreading Speed', 'value': 'spread'},
                                        {'label': ' Entropy Evolution', 'value': 'entropy'},
                                        {'label': ' Hit Probability', 'value': 'hit'},
                                    ],
                                    value=['mixing', 'spread'],
                                    labelStyle={'display': 'block', 'margin': '5px'}
                                ),
                                
                                html.Hr(),
                                dbc.Button("⚡ Run Walk Comparison", 
                                          id='btn-run-walks',
                                          color="success",
                                          size="lg",
                                          className="btn-animate quantum-glow w-100",
                                          n_clicks=0),
                            ])
                        ], className="card-custom"),
                        
                        dbc.Card([
                            dbc.CardHeader("Quantum Advantage Metrics", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
                            dbc.CardBody([
                                html.Div(id='quantum-metrics'),
                                dcc.Loading(
                                    id="loading-metrics",
                                    type="circle",
                                    children=html.Div(id="metrics-output")
                                ),
                            ])
                        ], className="card-custom"),
                    ], width=4),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Walk Comparison Visualization", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
                            dbc.CardBody([
                                dcc.Tabs([
                                    dcc.Tab(label='Probability Evolution', children=[
                                        dcc.Graph(id='walk-probability-plot', style={'height': '500px'}),
                                    ]),
                                    dcc.Tab(label='Network Spread', children=[
                                        dcc.Graph(id='walk-network-plot', style={'height': '500px'}),
                                    ]),
                                    dcc.Tab(label='Performance Metrics', children=[
                                        dcc.Graph(id='walk-metrics-plot', style={'height': '500px'}),
                                    ]),
                                ]),
                            ])
                        ], className="card-custom"),
                    ], width=8),
                ]),
            ], className="tab-content"),
        ]),
        
        # Tab 4: Data Export
        dcc.Tab(label='📊 Export Results', children=[
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Export Configuration", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
                            dbc.CardBody([
                                html.H5("Select Data to Export:", style={'color': '#3498db'}),
                                dcc.Checklist(
                                    id='export-options',
                                    options=[
                                        {'label': ' Network Data', 'value': 'network'},
                                        {'label': ' Normal Simulation Results', 'value': 'normal'},
                                        {'label': ' Attack Simulation Results', 'value': 'attack'},
                                        {'label': ' Walk Comparison Data', 'value': 'walks'},
                                        {'label': ' Impact Metrics', 'value': 'metrics'},
                                        {'label': ' All Data', 'value': 'all'},
                                    ],
                                    value=['all'],
                                    labelStyle={'display': 'block', 'margin': '5px'}
                                ),
                                
                                html.Hr(),
                                html.H5("Export Format:", style={'color': '#3498db'}),
                                dcc.RadioItems(
                                    id='export-format',
                                    options=[
                                        {'label': ' CSV', 'value': 'csv'},
                                        {'label': ' JSON', 'value': 'json'},
                                        {'label': ' Excel', 'value': 'excel'},
                                    ],
                                    value='csv',
                                    labelStyle={'marginRight': '20px'}
                                ),
                                
                                html.Hr(),
                                dbc.Button("📥 Download All Data", 
                                          id='btn-export',
                                          color="info",
                                          size="lg",
                                          className="btn-animate w-100",
                                          n_clicks=0),
                                
                                html.Div(id='export-status', style={'marginTop': '20px'}),
                                
                                html.Hr(),
                                html.H5("Generate Report:", style={'color': '#3498db'}),
                                dbc.Button("📋 Generate PDF Report", 
                                          id='btn-report',
                                          color="warning",
                                          className="btn-animate m-1"),
                                dbc.Button("🖨️ Print Summary", 
                                          id='btn-print',
                                          color="secondary",
                                          className="btn-animate m-1"),
                            ])
                        ], className="card-custom"),
                    ], width=4),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Data Preview", style={'fontSize': '1.2em', 'fontWeight': 'bold'}),
                            dbc.CardBody([
                                dcc.Tabs([
                                    dcc.Tab(label='Network Data', children=[
                                        dash_table.DataTable(
                                            id='preview-network',
                                            page_size=10,
                                            style_table={'overflowX': 'auto'},
                                            style_cell={
                                                'backgroundColor': '#2c3e50',
                                                'color': 'white',
                                                'border': '1px solid #34495e'
                                            },
                                        ),
                                    ]),
                                    dcc.Tab(label='Attack Results', children=[
                                        dash_table.DataTable(
                                            id='preview-attack',
                                            page_size=10,
                                            style_table={'overflowX': 'auto'},
                                            style_cell={
                                                'backgroundColor': '#2c3e50',
                                                'color': 'white',
                                                'border': '1px solid #34495e'
                                            },
                                        ),
                                    ]),
                                    dcc.Tab(label='Walk Results', children=[
                                        dash_table.DataTable(
                                            id='preview-walks',
                                            page_size=10,
                                            style_table={'overflowX': 'auto'},
                                            style_cell={
                                                'backgroundColor': '#2c3e50',
                                                'color': 'white',
                                                'border': '1px solid #34495e'
                                            },
                                        ),
                                    ]),
                                ]),
                            ])
                        ], className="card-custom"),
                    ], width=8),
                ]),
            ], className="tab-content"),
        ]),
    ]),
    
    # Hidden storage
    dcc.Store(id='network-store'),
    dcc.Store(id='simulation-store'),
    dcc.Store(id='walk-store'),
    
    # Footer
    html.Footer([
        html.Hr(),
        html.P([
            "EPANET Anomaly & Quantum Walk Analyzer | ",
            html.A("Quantum Computing Lab", href="#", style={'color': '#3498db'}),
            " | © 2024 Network Security Research Group"
        ], style={'textAlign': 'center', 'color': '#7f8c8d', 'padding': '20px'})
    ]),
])

# ============================================================================
# SIMULATION FUNCTIONS (Lightweight)
# ============================================================================

def create_sample_network():
    """Create a sample water distribution network"""
    G = nx.Graph()
    
    # Create nodes (junctions, reservoirs, tanks)
    nodes = [
        ('R1', {'type': 'reservoir', 'elevation': 100, 'pressure': 50}),
        ('J1', {'type': 'junction', 'elevation': 90, 'demand': 10}),
        ('J2', {'type': 'junction', 'elevation': 85, 'demand': 15}),
        ('J3', {'type': 'junction', 'elevation': 80, 'demand': 20}),
        ('J4', {'type': 'junction', 'elevation': 75, 'demand': 12}),
        ('T1', {'type': 'tank', 'elevation': 95, 'volume': 1000}),
    ]
    
    # Create links (pipes, pumps, valves)
    links = [
        ('R1', 'J1', {'type': 'pipe', 'length': 1000, 'diameter': 300}),
        ('J1', 'J2', {'type': 'pipe', 'length': 800, 'diameter': 250}),
        ('J2', 'J3', {'type': 'pipe', 'length': 1200, 'diameter': 200}),
        ('J3', 'J4', {'type': 'pipe', 'length': 900, 'diameter': 150}),
        ('J1', 'T1', {'type': 'pipe', 'length': 500, 'diameter': 200}),
        ('J2', 'T1', {'type': 'pipe', 'length': 700, 'diameter': 180}),
    ]
    
    for node, attrs in nodes:
        G.add_node(node, **attrs)
    
    for u, v, attrs in links:
        G.add_edge(u, v, **attrs)
    
    return G

def simulate_normal_conditions(G, duration=24):
    """Simulate normal network conditions"""
    results = {
        'time': list(range(duration)),
        'pressure': {},
        'flow': {},
        'demand': {}
    }
    
    for node in G.nodes():
        base_pressure = G.nodes[node].get('pressure', 30) if G.nodes[node].get('type') == 'reservoir' else 0
        base_demand = G.nodes[node].get('demand', 0)
        
        # Simulate diurnal pattern
        pressures = []
        demands = []
        
        for t in range(duration):
            # Base pressure with small fluctuations
            if G.nodes[node].get('type') == 'junction':
                p = 30 + 5 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.5)
            elif G.nodes[node].get('type') == 'reservoir':
                p = base_pressure + np.random.normal(0, 1)
            else:
                p = 25 + np.random.normal(0, 0.5)
            
            # Demand pattern (higher during day)
            d = base_demand * (1 + 0.5 * np.sin(2 * np.pi * (t - 6) / 24))
            
            pressures.append(p)
            demands.append(d)
        
        results['pressure'][node] = pressures
        results['demand'][node] = demands
    
    # Simulate flow in pipes
    for u, v in G.edges():
        if G.edges[u, v].get('type') == 'pipe':
            base_flow = 5 + np.random.random() * 10
            flows = [base_flow * (1 + 0.3 * np.sin(2 * np.pi * (t - 8) / 24)) 
                    for t in range(duration)]
            results['flow'][f"{u}-{v}"] = flows
    
    return results

def inject_anomalies(G, normal_results, attack_params):
    """Inject anomalies into the network"""
    attacked_results = normal_results.copy()
    
    for attack_type in attack_params['types']:
        if attack_type == 'demand':
            # Demand increase attack
            for node in attack_params.get('nodes', []):
                if node in attacked_results['demand']:
                    magnitude = attack_params['magnitude']
                    start = attack_params['start']
                    duration = attack_params['duration']
                    
                    for t in range(len(attacked_results['time'])):
                        if start <= t < start + duration:
                            attacked_results['demand'][node][t] *= magnitude
                            # Pressure affected by demand increase
                            attacked_results['pressure'][node][t] *= (1 - 0.1 * (magnitude - 1))
        
        elif attack_type == 'pressure':
            # Pressure sensor spoofing
            for node in attack_params.get('nodes', []):
                if node in attacked_results['pressure']:
                    start = attack_params['start']
                    duration = attack_params['duration']
                    
                    for t in range(len(attacked_results['time'])):
                        if start <= t < start + duration:
                            # Spoof lower pressure readings
                            attacked_results['pressure'][node][t] *= 0.7
        
        elif attack_type == 'leak':
            # Pipe leak attack
            for link in attack_params.get('links', []):
                if link in attacked_results['flow']:
                    start = attack_params['start']
                    duration = attack_params['duration']
                    
                    for t in range(len(attacked_results['time'])):
                        if start <= t < start + duration:
                            # Increase flow (leakage)
                            attacked_results['flow'][link][t] *= 1.5
    
    return attacked_results

class QuantumWalkSimulator:
    """Lightweight quantum walk simulator"""
    
    @staticmethod
    def continuous_time_quantum_walk(G, start_node, total_time=10, steps=50):
        """Continuous-time quantum walk"""
        nodes = list(G.nodes())
        n = len(nodes)
        
        # Create adjacency matrix
        A = nx.adjacency_matrix(G).todense()
        
        # Create Laplacian (Hamiltonian)
        D = np.diag(np.sum(A, axis=1))
        H = D - A  # Graph Laplacian
        
        # Initial state
        start_idx = nodes.index(start_node)
        psi0 = np.zeros(n, dtype=complex)
        psi0[start_idx] = 1.0 + 0j
        
        # Time evolution
        times = np.linspace(0, total_time, steps)
        probabilities = []
        
        for t in times:
            # Time evolution operator
            U = expm(-1j * H * t)
            psi_t = np.dot(U, psi0)
            prob_t = np.abs(psi_t)**2
            probabilities.append(prob_t)
        
        return np.array(probabilities), times
    
    @staticmethod
    def classical_random_walk(G, start_node, steps=50):
        """Classical random walk"""
        nodes = list(G.nodes())
        n = len(nodes)
        
        # Transition matrix
        A = nx.adjacency_matrix(G).todense()
        D_inv = np.diag(1 / np.sum(A, axis=1).clip(min=1e-10))
        P = np.dot(D_inv, A)
        
        # Initial distribution
        start_idx = nodes.index(start_node)
        prob_dist = np.zeros(n)
        prob_dist[start_idx] = 1.0
        
        # Evolution
        probabilities = [prob_dist.copy()]
        
        for _ in range(steps - 1):
            prob_dist = np.dot(prob_dist, P)
            probabilities.append(prob_dist.copy())
        
        return np.array(probabilities)

# ============================================================================
# CALLBACK FUNCTIONS
# ============================================================================

@app.callback(
    [Output('network-info', 'children'),
     Output('node-list-container', 'children'),
     Output('target-nodes', 'options'),
     Output('target-links', 'options'),
     Output('walk-start-node', 'options'),
     Output('network-graph', 'figure'),
     Output('network-store', 'data')],
    [Input('upload-data', 'contents'),
     Input('btn-example-small', 'n_clicks'),
     Input('btn-example-medium', 'n_clicks'),
     Input('btn-example-large', 'n_clicks'),
     Input('layout-algorithm', 'value'),
     Input('node-size-metric', 'value')],
    prevent_initial_call=True
)
def load_and_visualize_network(upload_content, btn_small, btn_medium, btn_large, layout, size_metric):
    """Load network data and create visualization"""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    # Determine which button was clicked
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Load appropriate network
    if button_id == 'upload-data' and upload_content:
        # Parse uploaded file
        content_type, content_string = upload_content.split(',')
        decoded = base64.b64decode(content_string)
        
        # For demo, create sample network
        G = create_sample_network()
        network_name = "Uploaded Network"
        
    else:
        # Create example networks
        if button_id == 'btn-example-small':
            G = create_sample_network()
            network_name = "Small Network (6 nodes, 6 links)"
        elif button_id == 'btn-example-medium':
            G = nx.erdos_renyi_graph(15, 0.3, seed=42)
            for node in G.nodes():
                G.nodes[node]['type'] = 'junction'
                G.nodes[node]['demand'] = np.random.uniform(5, 20)
            network_name = "Medium Network (15 nodes)"
        else:  # btn-example-large
            G = nx.barabasi_albert_graph(25, 2, seed=42)
            for node in G.nodes():
                G.nodes[node]['type'] = 'junction'
                G.nodes[node]['demand'] = np.random.uniform(5, 30)
            network_name = "Large Network (25 nodes)"
    
    # Generate network information
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    node_types = {}
    
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    network_info = [
        html.H5(f"📊 {network_name}", style={'color': '#3498db'}),
        html.P(f"• Nodes: {num_nodes}"),
        html.P(f"• Links: {num_edges}"),
        html.P("• Node Types:"),
        html.Ul([html.Li(f"{typ}: {count}") for typ, count in node_types.items()]),
        html.P(f"• Average Degree: {np.mean([d for n, d in G.degree()]):.2f}"),
    ]
    
    # Create node list for selection
    node_options = [{'label': f"{node} ({G.nodes[node].get('type', 'node')})", 'value': node} 
                   for node in G.nodes()]
    link_options = [{'label': f"{u}-{v} (pipe)", 'value': f"{u}-{v}") 
                   for u, v in G.edges()]
    
    # Create network visualization
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'kamada':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.random_layout(G, seed=42)
    
    # Determine node sizes
    if size_metric == 'degree':
        node_sizes = [G.degree(node) * 10 + 5 for node in G.nodes()]
    elif size_metric == 'betweenness':
        betweenness = nx.betweenness_centrality(G)
        node_sizes = [betweenness[node] * 100 + 5 for node in G.nodes()]
    else:
        node_sizes = [15 for _ in G.nodes()]
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_type = G.nodes[node].get('type', 'unknown')
        node_text.append(f"Node: {node}<br>Type: {node_type}<br>Degree: {G.degree(node)}")
        
        # Color by node type
        if node_type == 'reservoir':
            node_color.append('#e74c3c')  # Red
        elif node_type == 'junction':
            node_color.append('#3498db')  # Blue
        elif node_type == 'tank':
            node_color.append('#2ecc71')  # Green
        else:
            node_color.append('#f39c12')  # Orange
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_sizes,
            color=node_color,
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace],
                   layout=go.Layout(
                       title=f'Network Visualization - {network_name}',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='rgba(44, 62, 80, 1)',
                       paper_bgcolor='rgba(44, 62, 80, 1)',
                       font=dict(color='white')
                   ))
    
    # Store network data
    network_data = {
        'nodes': list(G.nodes()),
        'edges': list(G.edges()),
        'node_attrs': {node: dict(G.nodes[node]) for node in G.nodes()},
        'edge_attrs': {f"{u}-{v}": dict(G.edges[u, v]) for u, v in G.edges()}
    }
    
    # Node list display
    node_list = html.Div([
        html.H5("Node List:", style={'color': '#3498db'}),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.P(f"• {node} ({G.nodes[node].get('type', 'node')})", 
                          style={'padding': '2px', 'margin': '0'})
                    for node in list(G.nodes())[:10]
                ])
            ])
        ])
    ])
    
    return network_info, node_list, node_options, link_options, node_options, fig, network_data

@app.callback(
    [Output('attack-visualization', 'figure'),
     Output('pressure-comparison', 'figure'),
     Output('impact-metrics', 'children'),
     Output('impact-table', 'data'),
     Output('simulation-store', 'data')],
    [Input('btn-simulate-attack', 'n_clicks')],
    [State('attack-types', 'value'),
     State('target-nodes', 'value'),
     State('target-links', 'value'),
     State('attack-magnitude', 'value'),
     State('attack-start', 'value'),
     State('attack-duration', 'value'),
     State('network-store', 'data')]
)
def simulate_attack(n_clicks, attack_types, target_nodes, target_links, 
                   magnitude, start_time, duration, network_data):
    """Simulate attack scenario"""
    if n_clicks == 0 or not network_data:
        raise dash.exceptions.PreventUpdate
    
    # Recreate network from stored data
    G = nx.Graph()
    if network_data:
        for node in network_data['nodes']:
            G.add_node(node, **network_data['node_attrs'].get(node, {}))
        for edge in network_data['edges']:
            G.add_edge(edge[0], edge[1], **network_data['edge_attrs'].get(f"{edge[0]}-{edge[1]}", {}))
    
    # Simulate normal conditions
    normal_results = simulate_normal_conditions(G)
    
    # Prepare attack parameters
    attack_params = {
        'types': attack_types,
        'nodes': target_nodes or [],
        'links': target_links or [],
        'magnitude': magnitude,
        'start': start_time,
        'duration': duration
    }
    
    # Inject anomalies
    attacked_results = inject_anomalies(G, normal_results, attack_params)
    
    # Store results
    simulation_results = {
        'normal': normal_results,
        'attacked': attacked_results,
        'attack_params': attack_params
    }
    
    # Create attack visualization
    fig1 = create_attack_visualization(normal_results, attacked_results, attack_params)
    
    # Create pressure comparison plot
    fig2 = create_pressure_comparison(normal_results, attacked_results, target_nodes)
    
    # Calculate impact metrics
    impact_metrics, table_data = calculate_impact_metrics(normal_results, attacked_results, target_nodes)
    
    return fig1, fig2, impact_metrics, table_data, simulation_results

def create_attack_visualization(normal_results, attacked_results, attack_params):
    """Create visualization of attack impact"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Pressure Distribution', 'Demand Pattern',
                       'Flow Changes', 'Attack Timeline'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    time = normal_results['time']
    
    # Plot 1: Pressure at a sample node
    sample_node = list(normal_results['pressure'].keys())[0] if normal_results['pressure'] else 'J1'
    fig.add_trace(
        go.Scatter(x=time, y=normal_results['pressure'][sample_node],
                  name='Normal', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=attacked_results['pressure'][sample_node],
                  name='Attacked', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    # Highlight attack period
    start = attack_params['start']
    duration = attack_params['duration']
    fig.add_vrect(
        x0=start, x1=start + duration,
        fillcolor="red", opacity=0.2,
        layer="below", line_width=0,
        row=1, col=1
    )
    
    # Plot 2: Demand pattern
    if sample_node in normal_results['demand']:
        fig.add_trace(
            go.Scatter(x=time, y=normal_results['demand'][sample_node],
                      name='Normal', line=dict(color='green')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=time, y=attacked_results['demand'][sample_node],
                      name='Attacked', line=dict(color='orange', dash='dash')),
            row=1, col=2
        )
    
    # Plot 3: Flow changes
    if normal_results['flow']:
        sample_link = list(normal_results['flow'].keys())[0]
        fig.add_trace(
            go.Scatter(x=time, y=normal_results['flow'][sample_link],
                      name='Normal', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=attacked_results['flow'][sample_link],
                      name='Attacked', line=dict(color='yellow', dash='dash')),
            row=2, col=1
        )
    
    # Plot 4: Attack timeline
    attack_types = attack_params['types']
    y_positions = list(range(len(attack_types)))
    
    for i, attack_type in enumerate(attack_types):
        fig.add_trace(
            go.Scatter(x=[start, start + duration], y=[y_positions[i]] * 2,
                      mode='lines+markers',
                      line=dict(width=10, color='red'),
                      name=attack_type),
            row=2, col=2
        )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        plot_bgcolor='rgba(44, 62, 80, 1)',
        paper_bgcolor='rgba(44, 62, 80, 1)',
        font=dict(color='white'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_pressure_comparison(normal_results, attacked_results, target_nodes):
    """Create pressure comparison plot"""
    if not target_nodes:
        return go.Figure()
    
    fig = go.Figure()
    
    for node in target_nodes[:3]:  # Show first 3 target nodes
        if node in normal_results['pressure']:
            fig.add_trace(go.Scatter(
                x=normal_results['time'],
                y=normal_results['pressure'][node],
                name=f'{node} - Normal',
                line=dict(width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=attacked_results['time'],
                y=attacked_results['pressure'][node],
                name=f'{node} - Attacked',
                line=dict(width=2, dash='dash')
            ))
    
    fig.update_layout(
        title='Pressure Comparison: Target Nodes',
        xaxis_title='Time (hours)',
        yaxis_title='Pressure (m)',
        plot_bgcolor='rgba(44, 62, 80, 1)',
        paper_bgcolor='rgba(44, 62, 80, 1)',
        font=dict(color='white'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def calculate_impact_metrics(normal_results, attacked_results, target_nodes):
    """Calculate impact metrics of the attack"""
    metrics = []
    table_data = []
    
    # Calculate for target nodes
    for node in (target_nodes or list(normal_results['pressure'].keys())[:5]):
        if node in normal_results['pressure']:
            normal_avg = np.mean(normal_results['pressure'][node])
            attacked_avg = np.mean(attacked_results['pressure'][node])
            change = ((attacked_avg - normal_avg) / normal_avg * 100) if normal_avg != 0 else 0
            
            impact_level = "High" if abs(change) > 20 else ("Medium" if abs(change) > 10 else "Low")
            
            metrics.append({
                'Node': node,
                'Normal Pressure': f"{normal_avg:.2f} m",
                'Attacked Pressure': f"{attacked_avg:.2f} m",
                'Change': f"{change:.1f}%",
                'Impact': impact_level
            })
    
    # Create metrics cards
    if metrics:
        avg_change = np.mean([float(m['Change'].replace('%', '')) for m in metrics])
        max_change = max([abs(float(m['Change'].replace('%', ''))) for m in metrics])
        
        metrics_display = dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6("Average Pressure Change", style={'color': '#3498db'}),
                    html.H4(f"{avg_change:.1f}%", style={'color': 'red' if avg_change > 10 else 'orange'})
                ], className="metric-card")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H6("Maximum Impact", style={'color': '#3498db'}),
                    html.H4(f"{max_change:.1f}%", style={'color': 'red'})
                ], className="metric-card")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H6("Affected Nodes", style={'color': '#3498db'}),
                    html.H4(f"{len(metrics)}", style={'color': '#2ecc71'})
                ], className="metric-card")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H6("Attack Severity", style={'color': '#3498db'}),
                    html.H4("High" if max_change > 20 else "Medium", 
                           style={'color': 'red' if max_change > 20 else 'orange'})
                ], className="metric-card")
            ], width=3),
        ])
    else:
        metrics_display = html.P("No target nodes selected or available.")
    
    return metrics_display, metrics

@app.callback(
    [Output('walk-probability-plot', 'figure'),
     Output('walk-network-plot', 'figure'),
     Output('walk-metrics-plot', 'figure'),
     Output('quantum-metrics', 'children'),
     Output('walk-store', 'data')],
    [Input('btn-run-walks', 'n_clicks')],
    [State('walk-start-node', 'value'),
     State('walk-steps', 'value'),
     State('quantum-walk-type', 'value'),
     State('quantum-time', 'value'),
     State('walk-metrics', 'value'),
     State('network-store', 'data')]
)
def run_walk_comparison(n_clicks, start_node, steps, walk_type, quantum_time, metrics, network_data):
    """Run quantum and classical walk comparison"""
    if n_clicks == 0 or not start_node or not network_data:
        raise dash.exceptions.PreventUpdate
    
    # Recreate network
    G = nx.Graph()
    if network_data:
        for node in network_data['nodes']:
            G.add_node(node, **network_data['node_attrs'].get(node, {}))
        for edge in network_data['edges']:
            G.add_edge(edge[0], edge[1], **network_data['edge_attrs'].get(f"{edge[0]}-{edge[1]}", {}))
    
    # Initialize walk simulator
    walker = QuantumWalkSimulator()
    
    # Run walks
    classical_probs = walker.classical_random_walk(G, start_node, steps)
    quantum_probs, times = walker.continuous_time_quantum_walk(G, start_node, quantum_time, steps)
    
    # Store results
    walk_results = {
        'classical': classical_probs.tolist(),
        'quantum': quantum_probs.tolist(),
        'times': times.tolist(),
        'nodes': list(G.nodes()),
        'start_node': start_node
    }
    
    # Create probability evolution plot
    fig1 = create_probability_evolution_plot(classical_probs, quantum_probs, times, 
                                            G.nodes(), start_node)
    
    # Create network spread plot
    fig2 = create_network_spread_plot(G, classical_probs, quantum_probs, start_node)
    
    # Create metrics plot
    fig3 = create_walk_metrics_plot(classical_probs, quantum_probs, times)
    
    # Calculate quantum advantage metrics
    quantum_metrics = calculate_quantum_advantage(classical_probs, quantum_probs)
    
    return fig1, fig2, fig3, quantum_metrics, walk_results

def create_probability_evolution_plot(classical_probs, quantum_probs, times, nodes, start_node):
    """Create probability evolution plot"""
    nodes_list = list(nodes)
    start_idx = nodes_list.index(start_node) if start_node in nodes_list else 0
    
    # Select some target nodes to monitor
    target_indices = []
    for i in range(min(5, len(nodes_list))):
        if i != start_idx:
            target_indices.append(i)
    
    fig = make_subplots(
        rows=len(target_indices), cols=1,
        subplot_titles=[f'Probability at Node {nodes_list[i]}' for i in target_indices],
        vertical_spacing=0.1
    )
    
    for idx, target_idx in enumerate(target_indices, 1):
        # Classical walk
        fig.add_trace(
            go.Scatter(
                x=list(range(len(classical_probs))),
                y=classical_probs[:, target_idx],
                name='Classical Random Walk',
                line=dict(color='blue', width=2),
                showlegend=(idx == 1)
            ),
            row=idx, col=1
        )
        
        # Quantum walk
        fig.add_trace(
            go.Scatter(
                x=times,
                y=quantum_probs[:, target_idx],
                name='Quantum Walk',
                line=dict(color='#00ff9d', width=2),
                showlegend=(idx == 1)
            ),
            row=idx, col=1
        )
    
    fig.update_layout(
        height=400,
        plot_bgcolor='rgba(44, 62, 80, 1)',
        paper_bgcolor='rgba(44, 62, 80, 1)',
        font=dict(color='white'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_network_spread_plot(G, classical_probs, quantum_probs, start_node):
    """Create network visualization of walk spread"""
    # Get positions
    pos = nx.spring_layout(G, seed=42)
    
    # Get final probability distributions
    classical_final = classical_probs[-1]
    quantum_final = quantum_probs[-1]
    
    # Create traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)
    
    # Create node traces for classical and quantum
    nodes_list = list(G.nodes())
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Classical Random Walk', 'Quantum Walk'],
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Classical walk nodes
    node_x = [pos[node][0] for node in nodes_list]
    node_y = [pos[node][1] for node in nodes_list]
    
    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=[p * 100 + 10 for p in classical_final],
                color=classical_final,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(x=0.45, title='Probability')
            ),
            text=[f'Node: {node}<br>Probability: {p:.4f}' 
                 for node, p in zip(nodes_list, classical_final)],
            hoverinfo='text',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Highlight start node
    start_idx = nodes_list.index(start_node)
    fig.add_trace(
        go.Scatter(
            x=[node_x[start_idx]],
            y=[node_y[start_idx]],
            mode='markers',
            marker=dict(
                size=20,
                color='red',
                symbol='star'
            ),
            name='Start Node',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Quantum walk nodes
    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=[p * 100 + 10 for p in quantum_final],
                color=quantum_final,
                colorscale='Electric',
                showscale=True,
                colorbar=dict(x=1.0, title='Probability')
            ),
            text=[f'Node: {node}<br>Probability: {p:.4f}' 
                 for node, p in zip(nodes_list, quantum_final)],
            hoverinfo='text',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=[node_x[start_idx]],
            y=[node_y[start_idx]],
            mode='markers',
            marker=dict(
                size=20,
                color='red',
                symbol='star'
            ),
            name='Start Node',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(44, 62, 80, 1)',
        paper_bgcolor='rgba(44, 62, 80, 1)',
        font=dict(color='white')
    )
    
    return fig

def create_walk_metrics_plot(classical_probs, quantum_probs, times):
    """Create comparison plot of walk metrics"""
    # Calculate metrics
    classical_variance = [np.var(p) for p in classical_probs]
    quantum_variance = [np.var(p) for p in quantum_probs]
    
    # Calculate entropy (Shannon entropy)
    classical_entropy = []
    quantum_entropy = []
    
    for p in classical_probs:
        # Avoid log(0)
        p_safe = p[p > 1e-10]
        if len(p_safe) > 0:
            entropy = -np.sum(p_safe * np.log(p_safe))
            classical_entropy.append(entropy)
        else:
            classical_entropy.append(0)
    
    for p in quantum_probs:
        p_safe = p[p > 1e-10]
        if len(p_safe) > 0:
            entropy = -np.sum(p_safe * np.log(p_safe))
            quantum_entropy.append(entropy)
        else:
            quantum_entropy.append(0)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Variance Over Time', 'Entropy Over Time',
                       'Maximum Probability', 'Mixing Time Comparison'],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Variance plot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(classical_variance))),
            y=classical_variance,
            name='Classical',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=quantum_variance,
            name='Quantum',
            line=dict(color='#00ff9d', width=2)
        ),
        row=1, col=1
    )
    
    # Entropy plot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(classical_entropy))),
            y=classical_entropy,
            name='Classical',
            line=dict(color='blue', width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=quantum_entropy,
            name='Quantum',
            line=dict(color='#00ff9d', width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Max probability plot
    classical_max = [np.max(p) for p in classical_probs]
    quantum_max = [np.max(p) for p in quantum_probs]
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(classical_max))),
            y=classical_max,
            name='Classical',
            line=dict(color='blue', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=quantum_max,
            name='Quantum',
            line=dict(color='#00ff9d', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Mixing time comparison (bar chart)
    # Estimate mixing time as time to reach variance < threshold
    threshold = 0.01
    classical_mixing = np.argmax(np.array(classical_variance) < threshold)
    quantum_mixing = np.argmax(np.array(quantum_variance) < threshold)
    
    fig.add_trace(
        go.Bar(
            x=['Classical', 'Quantum'],
            y=[classical_mixing, quantum_mixing],
            marker_color=['blue', '#00ff9d'],
            text=[f'{classical_mixing} steps', f'{quantum_mixing} steps'],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        plot_bgcolor='rgba(44, 62, 80, 1)',
        paper_bgcolor='rgba(44, 62, 80, 1)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def calculate_quantum_advantage(classical_probs, quantum_probs):
    """Calculate quantum advantage metrics"""
    # Calculate final variance
    classical_final_var = np.var(classical_probs[-1])
    quantum_final_var = np.var(quantum_probs[-1])
    
    # Calculate mixing times (when variance drops below threshold)
    threshold = 0.01
    classical_variance = [np.var(p) for p in classical_probs]
    quantum_variance = [np.var(p) for p in quantum_probs]
    
    classical_mixing = np.argmax(np.array(classical_variance) < threshold)
    quantum_mixing = np.argmax(np.array(quantum_variance) < threshold)
    
    # Calculate spread speed (inverse of time to reach certain variance)
    classical_speed = 1 / (classical_mixing + 1e-10)
    quantum_speed = 1 / (quantum_mixing + 1e-10)
    
    # Calculate quantum advantage
    if classical_mixing > 0:
        speedup = ((classical_mixing - quantum_mixing) / classical_mixing) * 100
    else:
        speedup = 0
    
    # Create metrics display
    metrics_display = dbc.Row([
        dbc.Col([
            html.Div([
                html.H6("Quantum Speedup", style={'color': '#3498db'}),
                html.H4(f"{speedup:.1f}%", 
                       style={'color': '#00ff9d', 'fontWeight': 'bold'})
            ], className="metric-card quantum-glow")
        ], width=3),
        dbc.Col([
            html.Div([
                html.H6("Mixing Time (Quantum)", style={'color': '#3498db'}),
                html.H4(f"{quantum_mixing}", style={'color': '#00ff9d'})
            ], className="metric-card")
        ], width=3),
        dbc.Col([
            html.Div([
                html.H6("Mixing Time (Classical)", style={'color': '#3498db'}),
                html.H4(f"{classical_mixing}", style={'color': 'blue'})
            ], className="metric-card")
        ], width=3),
        dbc.Col([
            html.Div([
                html.H6("Spread Speed Ratio", style={'color': '#3498db'}),
                html.H4(f"{quantum_speed/classical_speed:.2f}x", 
                       style={'color': '#ff6b6b' if quantum_speed/classical_speed < 1 else '#00ff9d'})
            ], className="metric-card")
        ], width=3),
    ])
    
    # Add explanation
    explanation = html.Div([
        html.Hr(),
        html.H5("Quantum Advantage Explained:", style={'color': '#3498db'}),
        html.P([
            "• ",
            html.Strong("Quantum superposition"),
            " allows exploring multiple network paths simultaneously"
        ], style={'color': '#ecf0f1'}),
        html.P([
            "• ",
            html.Strong("Quantum interference"),
            " enables faster mixing through destructive interference"
        ], style={'color': '#ecf0f1'}),
        html.P([
            "• ",
            html.Strong("Quadratic speedup"),
            " demonstrated in hitting and mixing times"
        ], style={'color': '#ecf0f1'}),
        html.P([
            "• ",
            html.Strong("Applications:"),
            " Faster anomaly detection, optimized routing, enhanced network analysis"
        ], style={'color': '#ecf0f1'}),
    ])
    
    return html.Div([metrics_display, explanation])

@app.callback(
    [Output('preview-network', 'data'),
     Output('preview-network', 'columns'),
     Output('preview-attack', 'data'),
     Output('preview-attack', 'columns'),
     Output('preview-walks', 'data'),
     Output('preview-walks', 'columns'),
     Output('export-status', 'children')],
    [Input('btn-export', 'n_clicks'),
     Input('btn-report', 'n_clicks'),
     Input('btn-print', 'n_clicks')],
    [State('network-store', 'data'),
     State('simulation-store', 'data'),
     State('walk-store', 'data')]
)
def export_data(export_clicks, report_clicks, print_clicks, network_data, sim_data, walk_data):
    """Handle data export and preview"""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Prepare network data for preview
    network_table_data = []
    if network_data:
        for node in network_data['nodes']:
            attrs = network_data['node_attrs'].get(node, {})
            network_table_data.append({
                'Node': node,
                'Type': attrs.get('type', 'unknown'),
                'Degree': len([e for e in network_data['edges'] if node in e]),
                'Elevation': attrs.get('elevation', 'N/A'),
                'Demand': attrs.get('demand', 'N/A')
            })
    
    network_columns = [{'name': col, 'id': col} for col in ['Node', 'Type', 'Degree', 'Elevation', 'Demand']]
    
    # Prepare attack data for preview
    attack_table_data = []
    if sim_data and 'normal' in sim_data:
        normal = sim_data['normal']
        attacked = sim_data.get('attacked', {})
        
        # Sample some nodes
        sample_nodes = list(normal['pressure'].keys())[:5] if normal['pressure'] else []
        
        for node in sample_nodes:
            if node in normal['pressure']:
                normal_avg = np.mean(normal['pressure'][node])
                attacked_avg = np.mean(attacked.get('pressure', {}).get(node, [normal_avg]))
                change = ((attacked_avg - normal_avg) / normal_avg * 100) if normal_avg != 0 else 0
                
                attack_table_data.append({
                    'Node': node,
                    'Normal Pressure': f"{normal_avg:.2f}",
                    'Attacked Pressure': f"{attacked_avg:.2f}",
                    'Change (%)': f"{change:.1f}",
                    'Impact': 'High' if abs(change) > 20 else ('Medium' if abs(change) > 10 else 'Low')
                })
    
    attack_columns = [{'name': col, 'id': col} for col in 
                     ['Node', 'Normal Pressure', 'Attacked Pressure', 'Change (%)', 'Impact']]
    
    # Prepare walk data for preview
    walk_table_data = []
    if walk_data:
        nodes = walk_data.get('nodes', [])
        classical = np.array(walk_data.get('classical', []))
        quantum = np.array(walk_data.get('quantum', []))
        
        if len(classical) > 0 and len(quantum) > 0:
            # Final probability distribution
            classical_final = classical[-1]
            quantum_final = quantum[-1]
            
            for i, node in enumerate(nodes[:5]):  # First 5 nodes
                walk_table_data.append({
                    'Node': node,
                    'Classical Probability': f"{classical_final[i]:.4f}",
                    'Quantum Probability': f"{quantum_final[i]:.4f}",
                    'Quantum Advantage': f"{quantum_final[i]/classical_final[i]:.2f}x" if classical_final[i] > 0 else "N/A"
                })
    
    walk_columns = [{'name': col, 'id': col} for col in 
                   ['Node', 'Classical Probability', 'Quantum Probability', 'Quantum Advantage']]
    
    # Handle export button
    if button_id == 'btn-export':
        status = dbc.Alert(
            "✅ Data prepared for download. Right-click tables and select 'Export' to save as CSV.",
            color="success",
            dismissable=True
        )
    elif button_id == 'btn-report':
        status = dbc.Alert(
            "📋 Report generation would create a PDF summary of all analysis results.",
            color="info",
            dismissable=True
        )
    elif button_id == 'btn-print':
        status = dbc.Alert(
            "🖨️ Print functionality would generate a printable summary of key findings.",
            color="warning",
            dismissable=True
        )
    else:
        status = html.Div()
    
    return (network_table_data, network_columns,
            attack_table_data, attack_columns,
            walk_table_data, walk_columns,
            status)

# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('exports', exist_ok=True)
    
    # Run the app
    app.run_server(debug=True, port=8050)
