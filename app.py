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
st.sidebar.header("Population Parameters")

# Likert Toggle
use_likert = st.sidebar.checkbox("Discretize to Likert Scale", value=True, help="Uncheck for continuous scores (e.g. IQ)")

if use_likert:
    likert_points = st.sidebar.slider("Likert Scale Points", 2, 10, 5)
    mean_label = "Latent Mean (0=Avg)"
    std_label = "Latent SD (1=Std)"
else:
    likert_points = None
    mean_label = "Latent Mean (Target Avg)"
    std_label = "Latent SD (Target Std)"

pop_mean = st.sidebar.number_input(mean_label, value=0.0, step=0.1, format="%.2f")
pop_std = st.sidebar.number_input(std_label, value=1.0, step=0.1, format="%.2f")

# Warning for Likert configuration
if use_likert and abs(pop_mean) > 5.0:
    st.sidebar.warning("High Latent Mean with Likert enabled will cause ceiling effects. Uncheck 'Discretize' for continuous scores.")

st.sidebar.markdown("---")
st.sidebar.header("Sum Score Options")
calc_sum = st.sidebar.checkbox("Calculate Sum Score (Total)", value=False)
rescale_sum = st.sidebar.checkbox("Rescale Sum Score", value=False, disabled=not calc_sum)

if rescale_sum:
    col_t1, col_t2 = st.sidebar.columns(2)
    target_mean = col_t1.number_input("Target Mean", value=100.0)
    target_std = col_t2.number_input("Target SD", value=15.0)
    
    st.sidebar.markdown("**Rescaling Strategy**")
    rescale_strategy = st.sidebar.radio(
        "Strategy", 
        ["Current Sample (Auto)", "Use Fixed Reference (Comparison)"],
        label_visibility="collapsed"
    )
    
    if rescale_strategy == "Use Fixed Reference (Comparison)":
        if st.session_state.get("ref_raw_mean") is None:
            st.sidebar.error("No reference set! Generate a sample first and click 'Set Current Stats as Fixed Reference'.")
        else:
            st.sidebar.success(f"Ref: M={st.session_state.ref_raw_mean:.1f}, SD={st.session_state.ref_raw_std:.1f}")

st.sidebar.markdown("---")
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
    if st.button("Reset to Defaults"):
        st.session_state.loadings_df = None
        st.rerun()

    # Determine what to show
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
            # Pass new params
            df_result = generator.generate(n_samples=n_samples, likert_points=likert_points, mean=pop_mean, std=pop_std)
            
            # 5. Post-Processing (Sum Scores)
            if calc_sum:
                df_result['Total_Score'] = df_result.iloc[:, :n_vars].sum(axis=1) # Sum items only
                
                if rescale_sum:
                    # Determine Norms (Mean/SD to subtract/divide)
                    if rescale_strategy == "Use Fixed Reference (Comparison)" and st.session_state.get("ref_raw_mean") is not None:
                        current_mean = st.session_state.ref_raw_mean
                        current_std = st.session_state.ref_raw_std
                        st.info(f"Using Fixed Reference Norms: Mean={current_mean:.2f}, SD={current_std:.2f}")
                    else:
                        # Auto / Current Sample
                        current_mean = df_result['Total_Score'].mean()
                        current_std = df_result['Total_Score'].std()
                        # Save these for potential future reference
                        st.session_state.last_raw_mean = current_mean
                        st.session_state.last_raw_std = current_std
                    
                    # Z-score normalization using the determined norms
                    z_scores = (df_result['Total_Score'] - current_mean) / current_std
                    
                    df_result['Scaled_Score'] = z_scores * target_std + target_mean
                    df_result['Scaled_Score'] = df_result['Scaled_Score'].round(2)

            # 6. Success UI
            st.success(f"Generated {n_samples} subjects successfully!")
            
            # 7. Helper to set Reference
            # Logic Change: Button needs to persist the state. 
            # In Streamlit, buttons return True only on the click run.
            # We want to allow setting it from connection 1.
            
            if rescale_sum and rescale_strategy == "Current Sample (Auto)":
                if st.button("❄️ Set Current Stats as Fixed Reference"):
                    st.session_state.ref_raw_mean = df_result['Total_Score'].mean()
                    st.session_state.ref_raw_std = df_result['Total_Score'].std()
                    # Rerun to update sidebar
                    st.rerun()

            if rescale_sum and rescale_strategy == "Use Fixed Reference (Comparison)" and st.button("Reset Reference"):
                 st.session_state.ref_raw_mean = None
                 st.session_state.ref_raw_std = None
                 st.rerun()
            
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



