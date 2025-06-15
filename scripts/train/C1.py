import streamlit as st
import torch
import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, Qwen2ForCausalLM
import numpy as np

# Select device: MPS (Apple Silicon), CUDA, or CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

@st.cache_resource
def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # Use half precision on MPS/CUDA for faster inference
    dtype = torch.float16 if DEVICE.type in ["mps", "cuda"] else torch.float32
    model = Qwen2ForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": DEVICE.type},
        use_cache=True,
    )
    model.eval()
    # Optional compilation for PyTorch 2.0+
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    return tokenizer, model

def create_argument_graph(json_data):
    """Create an interactive argument graph using plotly and networkx"""
    try:
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for edu in data['edus']:
            G.add_node(edu['id'], text=edu['text'])
        
        # Add edges
        for relation in data['relations']:
            G.add_edge(relation['from'], relation['to'], relation_type=relation['type'])
        
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
            'Conclusion': '#34495e'
        }
        
        # Get roles for coloring
        roles_dict = {role['id']: role['role'] for role in data['roles']}
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Truncate text for display
            text = data['edus'][node]['text']
            display_text = text[:60] + "..." if len(text) > 60 else text
            node_text.append(f"ID {node}: {display_text}")
            
            # Color by role
            role = roles_dict.get(node, 'Unknown')
            node_colors.append(role_colors.get(role, '#95a5a6'))
            
            # Size by number of connections
            connections = len(list(G.neighbors(node))) + len(list(G.predecessors(node)))
            node_sizes.append(max(20, connections * 5))
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            relation_type = G.edges[edge]['relation_type']
            edge_info.append(f"{edge[0]} → {edge[1]}: {relation_type}")
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#7f8c8d'),
            hoverinfo='none',
            mode='lines',
            name='Relations'
        ))
        
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
            annotations=[ dict(
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
        return None, None

def display_argument_analysis(data):
    """Display detailed argument analysis"""
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.columns(3)
    
    with tab1:
        st.subheader("📊 Role Distribution")
        roles = [role['role'] for role in data['roles']]
        role_counts = {role: roles.count(role) for role in set(roles)}
        
        fig_pie = px.pie(
            values=list(role_counts.values()), 
            names=list(role_counts.keys()),
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.subheader("🎯 Stance Analysis")
        stances = data.get('stance', [])
        if stances:
            stance_df = []
            for stance in stances:
                stance_df.append({
                    'ID': stance['id'],
                    'Position': stance['position'],
                    'Strength': stance['modality'],
                    'Target': stance['target']
                })
            st.dataframe(stance_df, use_container_width=True)
        else:
            st.info("No stance data available")
    
    with tab3:
        st.subheader("🔗 Relations Summary")
        relations = data['relations']
        relation_types = [rel['type'] for rel in relations]
        relation_counts = {rel_type: relation_types.count(rel_type) for rel_type in set(relation_types)}
        
        for rel_type, count in relation_counts.items():
            st.metric(rel_type.replace('_', ' ').title(), count)

# Sidebar settings
st.sidebar.title("Settings")
model_dir = st.sidebar.text_input("Model directory", value="models/deepseek-argumentanalyst-full")
max_new_tokens = st.sidebar.number_input("Max new tokens", min_value=10, max_value=1020, value=200)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1)

# Graph settings
st.sidebar.subheader("Graph Settings")
show_analysis = st.sidebar.checkbox("Show detailed analysis", value=True)
show_raw_json = st.sidebar.checkbox("Show raw JSON", value=False)
debug_mode = st.sidebar.checkbox("Debug mode", value=False)

st.title("🧠 DeepSeek Argumentative Analysis")

# Load model
try:
    tokenizer, model = load_model(model_dir)
    st.success(f"✅ Model loaded successfully on {DEVICE}")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

prompt = st.text_area("Enter your argument prompt:", height=150, 
                     placeholder="Enter text to analyze its argumentative structure...")

# Add manual JSON input option
st.subheader("Manual JSON Input (Optional)")
manual_json = st.text_area("Or paste JSON directly:", height=100, 
                          placeholder='{"edus": [...], "roles": [...], "relations": [...]}')

if st.button("🚀 Generate Analysis", type="primary"):
    if manual_json.strip():
        # Use manual JSON input
        json_str = manual_json.strip()
        st.info("Using manually provided JSON")
        
        # Display results
        st.subheader("📈 Argument Graph")
        
        # Create and display graph
        fig, parsed_data = create_argument_graph(json_str)
        
        if fig and parsed_data:
            st.plotly_chart(fig, use_container_width=True)
            
            if show_analysis:
                st.subheader("📋 Detailed Analysis")
                display_argument_analysis(parsed_data)
            
            # Display argument components
            st.subheader("🎭 Argument Components")
            
            for i, edu in enumerate(parsed_data['edus']):
                role = next((r['role'] for r in parsed_data['roles'] if r['id'] == i), 'Unknown')
                
                # Color code by role
                if role == 'Claim':
                    st.error(f"**{role}** (ID {i}): {edu['text']}")
                elif role == 'Evidence':
                    st.success(f"**{role}** (ID {i}): {edu['text']}")
                elif role == 'Counterclaim':
                    st.warning(f"**{role}** (ID {i}): {edu['text']}")
                else:
                    st.info(f"**{role}** (ID {i}): {edu['text']}")
            
            if show_raw_json:
                st.subheader("🔍 Raw JSON Output")
                st.code(json_str, language="json")
        else:
            st.error("Failed to parse the provided JSON.")
            
    elif not prompt.strip():
        st.warning("Please enter a prompt to analyze or provide JSON directly.")
    else:
        with st.spinner("Analyzing argument structure..."):
            # Prepare input
            formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)
            
            # Generation under inference mode for speed
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON more robustly
            # Find the assistant's response part
            assistant_start = decoded.find('<|im_start|>assistant\n')
            if assistant_start != -1:
                response_text = decoded[assistant_start + len('<|im_start|>assistant\n'):]
            else:
                response_text = decoded
            
            # Remove any trailing assistant tags or artifacts
            response_text = response_text.replace('<|im_end|>', '').strip()
            
            # Find JSON boundaries
            json_str = response_text
            start_idx = response_text.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = len(response_text)
                
                for i, ch in enumerate(response_text[start_idx:], start=start_idx):
                    if ch == '{':
                        brace_count += 1
                    elif ch == '}':
                        brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
                
                json_str = response_text[start_idx:end_idx]
                
                # If JSON seems incomplete, try to fix common issues
                if not json_str.endswith('}'):
                    # Try to find where it should end and add missing braces
                    if '"stance":[' in json_str and not json_str.rstrip().endswith(']}'):
                        # Find the last complete entry and add closing braces
                        last_brace = json_str.rfind('}')
                        if last_brace != -1:
                            json_str = json_str[:last_brace+1] + ']}'
            
            # Validate and attempt to parse JSON
            try:
                test_parse = json.loads(json_str)
            except json.JSONDecodeError as e:
                # If parsing fails, try to fix the JSON
                st.warning(f"JSON parsing issue detected. Attempting to fix...")
                
                # Common fixes for incomplete JSON
                if not json_str.strip().endswith('}'):
                    if '"stance":[' in json_str:
                        # Find the last complete stance entry
                        lines = json_str.split('\n')
                        fixed_lines = []
                        in_stance = False
                        
                        for line in lines:
                            if '"stance":[' in line:
                                in_stance = True
                            
                            if in_stance and line.strip().endswith('}') and not line.strip().endswith(']}'):
                                # This might be the last stance entry
                                fixed_lines.append(line)
                                fixed_lines.append(']}')
                                break
                            else:
                                fixed_lines.append(line)
                        
                        json_str = '\n'.join(fixed_lines)
                
                # Try parsing again
                try:
                    test_parse = json.loads(json_str)
                except json.JSONDecodeError:
                    st.error("Could not parse JSON even after attempting fixes.")
                    st.code(json_str, language="json")
                    st.stop()
            
            # Display results
            if debug_mode:
                st.subheader("🔧 Debug Information")
                st.text_area("Raw Model Output", decoded, height=200)
                st.text_area("Extracted JSON", json_str, height=200)
            
            st.subheader("📈 Argument Graph")
            
            # Create and display graph
            fig, parsed_data = create_argument_graph(json_str)
            
            if fig and parsed_data:
                st.plotly_chart(fig, use_container_width=True)
                
                if show_analysis:
                    st.subheader("📋 Detailed Analysis")
                    display_argument_analysis(parsed_data)
                
                # Display argument components
                st.subheader("🎭 Argument Components")
                
                for i, edu in enumerate(parsed_data['edus']):
                    role = next((r['role'] for r in parsed_data['roles'] if r['id'] == i), 'Unknown')
                    
                    # Color code by role
                    if role == 'Claim':
                        st.error(f"**{role}** (ID {i}): {edu['text']}")
                    elif role == 'Evidence':
                        st.success(f"**{role}** (ID {i}): {edu['text']}")
                    elif role == 'Counterclaim':
                        st.warning(f"**{role}** (ID {i}): {edu['text']}")
                    else:
                        st.info(f"**{role}** (ID {i}): {edu['text']}")
                
                if show_raw_json:
                    st.subheader("🔍 Raw JSON Output")
                    st.code(json_str, language="json")
            
            else:
                st.error("Failed to parse the generated JSON. Please check the model output.")
                st.code(json_str, language="json")