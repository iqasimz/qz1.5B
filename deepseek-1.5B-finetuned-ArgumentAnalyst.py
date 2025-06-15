import os
import sys
import warnings

# More comprehensive environment setup to avoid torch watcher issues
os.environ["STREAMLIT_WATCHER_IGNORE"] = "torch,torch.*,transformers,transformers.*"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# Suppress all torch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")

import streamlit as st
import torch
import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, Qwen2ForCausalLM
import numpy as np

# Streamlit page config - must be first streamlit command
st.set_page_config(
    page_title="DeepSeek Argument Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Select device: MPS (Apple Silicon), CUDA, or CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

@st.cache_resource(show_spinner=True)
def load_model(model_dir: str):
    """Load the model with proper error handling and caching"""
    try:
        with st.spinner(f"Loading tokenizer from {model_dir}..."):
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        # Use half precision on MPS/CUDA for faster inference
        dtype = torch.float16 if DEVICE.type in ["mps", "cuda"] else torch.float32
        
        with st.spinner(f"Loading model on {DEVICE}..."):
            model = Qwen2ForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map={"": DEVICE.type},
                use_cache=True,
            )
            model.eval()
        
        # Optional compilation for PyTorch 2.0+
        if hasattr(torch, "compile") and DEVICE.type != "mps":  # Skip compile on MPS
            try:
                with st.spinner("Compiling model for optimization..."):
                    model = torch.compile(model)
            except Exception as e:
                st.warning(f"Model compilation failed, continuing without optimization: {e}")
        
        return tokenizer, model
    
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.info("Make sure the model directory is correct and accessible.")
        return None, None

def create_argument_graph(json_data):
    """Create an interactive argument graph using plotly and networkx"""
    try:
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        
        # Validate required fields
        if 'edus' not in data or 'relations' not in data:
            st.error("Invalid JSON structure: missing 'edus' or 'relations'")
            return None, None
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for i, edu in enumerate(data['edus']):
            G.add_node(i, text=edu['text'])
        
        # Add edges
        for relation in data['relations']:
            if 'from' in relation and 'to' in relation:
                G.add_edge(relation['from'], relation['to'], 
                          relation_type=relation.get('type', 'unknown'))
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        # Color mapping for roles
        role_colors = {
            'Claim': '#e74c3c',
            'Evidence': '#27ae60', 
            'Counterclaim': '#f39c12',
            'Elaboration': '#3498db',
            'Rebuttal': '#9b59b6',
            'Conclusion': '#34495e',
            'Unknown': '#95a5a6'
        }
        
        # Get roles for coloring
        roles_dict = {}
        if 'roles' in data:
            roles_dict = {role['id']: role['role'] for role in data['roles']}
        
        for node in G.nodes():
            if node not in pos:
                continue
                
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node text
            if node < len(data['edus']):
                text = data['edus'][node]['text']
                display_text = text[:60] + "..." if len(text) > 60 else text
                node_text.append(f"ID {node}: {display_text}")
            else:
                node_text.append(f"ID {node}: [No text]")
            
            # Color by role
            role = roles_dict.get(node, 'Unknown')
            node_colors.append(role_colors.get(role, '#95a5a6'))
            
            # Size by number of connections
            connections = len(list(G.neighbors(node))) + len(list(G.predecessors(node)))
            node_sizes.append(max(20, connections * 5 + 15))
        
        # Create edge traces
        edge_traces = []
        
        for edge in G.edges():
            if edge[0] not in pos or edge[1] not in pos:
                continue
                
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            relation_type = G.edges[edge].get('relation_type', 'unknown')
            
            # Add edge line
            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='#7f8c8d'),
                hoverinfo='none',
                showlegend=False
            ))
            
            # Add relation label at midpoint
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            
            edge_traces.append(go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode='text',
                text=[relation_type.replace('_', ' ')],
                textfont=dict(size=10, color='#2c3e50'),
                textposition="middle center",
                hoverinfo='text',
                hovertext=f"{edge[0]} → {edge[1]}: {relation_type}",
                showlegend=False
            ))
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add all edge traces
        for trace in edge_traces:
            fig.add_trace(trace)
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_text,
            text=[f"ID {i}" for i in range(len(node_x))],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            name='Arguments'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Argument Graph Visualization",
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[dict(
                text="Hover over nodes to see argument text",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600
        )
        
        return fig, data
        
    except Exception as e:
        st.error(f"Error creating graph: {str(e)}")
        st.code(str(json_data), language="text")
        return None, None

def display_argument_analysis(data):
    """Display detailed argument analysis"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎭 Argument Roles")
        if 'roles' in data and data['roles']:
            role_counts = {}
            for role in data['roles']:
                role_name = role.get('role', 'Unknown')
                role_counts[role_name] = role_counts.get(role_name, 0) + 1
            
            for role, count in role_counts.items():
                st.metric(role, count)
        else:
            st.info("No role information available")
    
    with col2:
        st.subheader("🎯 Stance Analysis")
        if 'stance' in data and data['stance']:
            stance_df = []
            for stance in data['stance']:
                stance_df.append({
                    'ID': stance.get('id', 'N/A'),
                    'Position': stance.get('position', 'N/A'),
                    'Strength': stance.get('modality', 'N/A'),
                    'Target': stance.get('target', 'N/A')
                })
            st.dataframe(stance_df, use_container_width=True)
        else:
            st.info("No stance data available")

def fix_incomplete_json(json_str):
    """Attempt to fix common JSON parsing issues"""
    json_str = json_str.strip()
    
    # Remove any trailing commas
    json_str = json_str.replace(',}', '}').replace(',]', ']')
    
    # If JSON doesn't end properly, try to close it
    if not json_str.endswith('}'):
        # Count open/close braces
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        
        if open_braces > close_braces:
            # Add missing closing braces
            json_str += '}' * (open_braces - close_braces)
    
    return json_str

# Main app
def main():
    st.title("🧠 DeepSeek Argumentative Analysis")
    st.markdown("Analyze the argumentative structure of text using AI")
    
    # Sidebar settings
    with st.sidebar:
        st.title("⚙️ Settings")
        
        model_dir = st.text_input(
            "Model directory", 
            value="iqasimz/deepseek-1.5B-argumentanalyst",
            help="HuggingFace model identifier or local path"
        )
        
        st.subheader("Generation Parameters")
        max_new_tokens = st.number_input("Max new tokens", min_value=10, max_value=1020, value=200)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
        
        st.subheader("Display Options")
        show_analysis = st.checkbox("Show detailed analysis", value=True)
        show_raw_json = st.checkbox("Show raw JSON", value=False)
        
        # Device info
        st.info(f"🖥️ Device: {DEVICE}")
    
    # Load model
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if not st.session_state.model_loaded:
        tokenizer, model = load_model(model_dir)
        if tokenizer is not None and model is not None:
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.success(f"✅ Model loaded successfully on {DEVICE}")
            st.rerun()
        else:
            st.error("❌ Failed to load model. Please check the model directory.")
            st.stop()
    
    # Input section
    st.subheader("📝 Input Text")
    prompt = st.text_area(
        "Enter your argument prompt:", 
        height=150, 
        placeholder="Enter text to analyze its argumentative structure...",
        help="Paste or type the text you want to analyze for arguments, claims, and evidence."
    )
    
    # Analysis button
    if st.button("🚀 Generate Analysis", type="primary", disabled=not prompt.strip()):
        if not prompt.strip():
            st.warning("Please enter a prompt to analyze.")
        else:
            with st.spinner("🔍 Analyzing argument structure..."):
                try:
                    # Prepare input
                    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    inputs = st.session_state.tokenizer(formatted, return_tensors="pt").to(DEVICE)
                    
                    # Generation
                    with torch.inference_mode():
                        outputs = st.session_state.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            pad_token_id=st.session_state.tokenizer.pad_token_id,
                            use_cache=True,
                        )
                    
                    decoded = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract JSON
                    assistant_start = decoded.find('<|im_start|>assistant\n')
                    if assistant_start != -1:
                        response_text = decoded[assistant_start + len('<|im_start|>assistant\n'):]
                    else:
                        response_text = decoded
                    
                    response_text = response_text.replace('<|im_end|>', '').strip()
                    
                    # Find JSON boundaries
                    start_idx = response_text.find('{')
                    if start_idx == -1:
                        st.error("No JSON found in model output")
                        st.code(response_text, language="text")
                        st.stop()
                    
                    json_str = response_text[start_idx:]
                    
                    # Try to find the end of JSON
                    brace_count = 0
                    end_idx = len(json_str)
                    
                    for i, ch in enumerate(json_str):
                        if ch == '{':
                            brace_count += 1
                        elif ch == '}':
                            brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                    
                    json_str = json_str[:end_idx]
                    
                    # Try to parse JSON
                    try:
                        parsed_data = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        st.warning("JSON parsing failed, attempting to fix...")
                        json_str = fix_incomplete_json(json_str)
                        try:
                            parsed_data = json.loads(json_str)
                        except json.JSONDecodeError:
                            st.error("Could not parse JSON output from model")
                            st.code(json_str, language="json")
                            st.stop()
                    
                    # Display results
                    st.success("✅ Analysis completed!")
                    
                    # Create tabs for better organization
                    tab1, tab2, tab3 = st.tabs(["📈 Graph", "📋 Analysis", "🔍 Raw Data"])
                    
                    with tab1:
                        fig, _ = create_argument_graph(parsed_data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Failed to create graph visualization")
                    
                    with tab2:
                        if show_analysis:
                            display_argument_analysis(parsed_data)
                        
                        # Display argument components
                        st.subheader("🎭 Argument Components")
                        
                        if 'edus' in parsed_data:
                            roles_dict = {}
                            if 'roles' in parsed_data:
                                roles_dict = {role['id']: role['role'] for role in parsed_data['roles']}
                            
                            for i, edu in enumerate(parsed_data['edus']):
                                role = roles_dict.get(i, 'Unknown')
                                
                                # Color code by role
                                if role == 'Claim':
                                    st.error(f"**{role}** (ID {i}): {edu['text']}")
                                elif role == 'Evidence':
                                    st.success(f"**{role}** (ID {i}): {edu['text']}")
                                elif role == 'Counterclaim':
                                    st.warning(f"**{role}** (ID {i}): {edu['text']}")
                                else:
                                    st.info(f"**{role}** (ID {i}): {edu['text']}")
                    
                    with tab3:
                        if show_raw_json:
                            st.subheader("🔍 Raw JSON Output")
                            st.code(json.dumps(parsed_data, indent=2), language="json")
                        
                        st.subheader("📄 Full Model Response")
                        st.code(response_text, language="text")
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)

if __name__ == "__main__":
    main()