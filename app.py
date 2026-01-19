import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import FactorModel, PopulationGenerator
from src.utils import create_simple_structure

st.set_page_config(page_title="Psychometric Generator", layout="wide")

st.title("Psychometric Population Generator")

# --- Sidebar Controls ---
st.sidebar.header("Global Settings")

structure_str = st.sidebar.text_input("Factor Structure (items per factor)", value="5,5,5", help="e.g., '5,10,5' for 3 factors")
reliability = st.sidebar.slider("Target Reliability", 0.0, 1.0, 0.8, 0.01)
n_samples = st.sidebar.number_input("Number of Subjects (N)", value=1000, step=100)

st.sidebar.markdown("---")
st.sidebar.header("Advanced Settings")
likert_points = st.sidebar.slider("Likert Scale Points", 2, 10, 5)
noise_level = st.sidebar.slider("Loading Noise", 0.0, 1.0, 0.0, 0.01, help="Random noise added to the factor loadings")

# --- Main Area ---

# Parse Structure
try:
    items_per_factor = [int(x.strip()) for x in structure_str.split(',')]
    n_factors = len(items_per_factor)
    n_vars = sum(items_per_factor)
except ValueError:
    st.error("Invalid Structure format. Please enter comma-separated integers (e.g., 5,5,5).")
    st.stop()

# --- Loadings Editor Logic ---

# Initialize session state for loadings if not present or if structure changed
if "loadings_df" not in st.session_state:
    st.session_state.loadings_df = None
if "last_structure" not in st.session_state:
    st.session_state.last_structure = structure_str

# Detect structure change to reset
if st.session_state.last_structure != structure_str:
    st.session_state.loadings_df = None
    st.session_state.last_structure = structure_str

def get_default_loadings():
    # Generate base structure
    defaults = create_simple_structure(n_factors, items_per_factor, reliability)
    
    # Apply Noise if specified (Visual only, realtime update)
    if noise_level > 0:
        noise_matrix = np.random.uniform(-noise_level, noise_level, size=defaults.shape)
        defaults = defaults + noise_matrix
        defaults = np.clip(defaults, -1.0, 1.0)
        
    return pd.DataFrame(
        defaults,
        columns=[f"Factor {i+1}" for i in range(n_factors)],
        index=[f"Item {i+1}" for i in range(n_vars)]
    )

st.subheader("Factor Loadings")
col1, col2 = st.columns([3, 1])

with col2:
    if st.button("Add Noise..."):
        st.session_state.loadings_df = None
        st.rerun()

# Determine what to show: User modified vs Calculated Defaults
# If we have a stored state, use it. If not, generate new defaults.
# Note: Changing sliders (Reliability/Noise) should technically update the view 
# UNLESS the user has actively edited specific cells. 
# For simplicity in this port: "Reset" forces a refresh from sliders. 
# Otherwise, we stick to what is in session state if it exists, 
# BUT we want slider changes to reflect immediately if the user hasn't "locked in" custom edits.
# A simpler approach for Streamlit: Just re-generate defaults every run UNLESS we are in a special "Custom Mode".
# But st.data_editor allows editing.

if st.session_state.loadings_df is None:
    # Fresh start or reset
    df_to_show = get_default_loadings()
else:
    # Use stored
    df_to_show = st.session_state.loadings_df

# Editor
edited_df = st.data_editor(df_to_show, height=400, use_container_width=True)

# Update state if changed
st.session_state.loadings_df = edited_df

# --- Generation ---
st.markdown("---")
if st.button("Generate Data", type="primary"):
    with st.spinner("Generating population..."):
        try:
            # 1. Get Loadings from Editor
            loadings_matrix = edited_df.to_numpy()
            
            # 2. Correlations
            phi = None
            if n_factors > 1:
                phi = np.eye(n_factors)
                phi[phi == 0] = 0.3
            
            # 3. Model
            model = FactorModel(loadings=loadings_matrix, factor_correlations=phi)
            
            # 4. Generate
            generator = PopulationGenerator(model)
            df_result = generator.generate(n_samples=n_samples, likert_points=likert_points)
            
            # 5. Success UI
            st.success(f"Generated {n_samples} subjects successfully!")
            
            # Preview
            st.dataframe(df_result.head(), use_container_width=True)
            
            # Download
            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="population_data.csv",
                mime="text/csv",
            )
            
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")

