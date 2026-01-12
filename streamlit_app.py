import streamlit as st
import torch
import numpy as np
import os
import sys
import json
import re

# Add tacrolimus-service to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tacrolimus-service'))

try:
    from timeseries_models import RNNModel
except ImportError:
    st.error("Could not import RNNModel. Make sure 'tacrolimus-service' directory exists and contains 'timeseries_models.py'.")

# --- Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Checkpoints are in tacrolimus-service/checkpoints
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'tacrolimus-service', 'checkpoints')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Helpers (Ported from predict_cli.py) ---

def parse_model_filename(filename):
    """Extract parameters from filename"""
    use_static = 'static' in filename
    
    if '_lstm_' in filename or filename.startswith('am_lstm') or filename.startswith('pm_lstm'):
        rnn_type = 'lstm'
    elif '_gru_' in filename or filename.startswith('am_gru') or filename.startswith('pm_gru'):
        rnn_type = 'gru'
    else:
        rnn_type = 'lstm'
    
    hd_match = re.search(r'hd(\d+)', filename)
    hidden_dim = int(hd_match.group(1)) if hd_match else 64
    
    nl_match = re.search(r'nl(\d+)', filename)
    num_layers = int(nl_match.group(1)) if nl_match else 2
    
    msl_match = re.search(r'msl(\d+)', filename)
    max_seq_len = int(msl_match.group(1)) if msl_match else 10
    
    return {
        'use_static': use_static,
        'rnn_type': rnn_type,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'max_seq_len': max_seq_len
    }

@st.cache_resource
def load_lstm_model(model_type, use_static, checkpoint_dir):
    """RNN Model Loader (Cached)"""
    try:
        if not os.path.exists(checkpoint_dir):
             return None, None, f"Checkpoint directory not found: {checkpoint_dir}"

        all_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        model_files = [f for f in all_files 
                      if f.startswith(f'{model_type}_lstm') or f.startswith(f'{model_type}_gru')]
        
        if use_static:
            model_files = [f for f in model_files if 'static' in f]
        else:
            model_files = [f for f in model_files if 'static' not in f]
        
        if not model_files:
            return None, None, f"{model_type.upper()} {'static' if use_static else 'non-static'} model file not found in {checkpoint_dir}"
        
        # Pick the most recent or first one. Let's pick the first one matching logic
        model_file = model_files[0]
        model_path = os.path.join(checkpoint_dir, model_file)
        
        params = parse_model_filename(model_file)
        rnn_type = params.get('rnn_type', 'lstm')
        
        if model_type == 'pm':
            input_dim = 3
        else:
            input_dim = 4
        
        static_dim = 10 if use_static else 0
        
        model = RNNModel(
            input_dim=input_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            use_static=use_static,
            static_dim=static_dim,
            rnn_type=rnn_type
        )
        
        # Load state dict
        # Map location to CPU is safer for Streamlit Cloud
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        return model, params, None
        
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

def round_prediction(value):
    if value < 0:
        return 0
    else:
        return round(value * 2) / 2

# --- UI Layout ---
st.set_page_config(page_title="KASAN AI Lab: IS Dose Prediction", page_icon="💊")

# Custom CSS for branding
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }
    h1 {
        color: #e65100;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("💊 KASAN AI Lab")
st.subheader("Immunosuppressant (Tacrolimus) Dose Prediction")

with st.expander("ℹ️  Instructions", expanded=False):
    st.write("Enter the patient's clinical indicators and most recent dosing information to predict the next recommended dosage.")

# Input Form
with st.form("prediction_form"):
    st.write("### 👤 Clinical Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=45)
        sex = st.radio("Sex", ["Male", "Female"])
        bwt = st.number_input("Weight (kg)", min_value=10.0, max_value=200.0, value=65.0)
        ht = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
        
        # Calculation for BMI
        bmi = bwt / ((ht/100) ** 2) if ht > 0 else 0
        st.info(f"Calculated BMI: {bmi:.2f}")

    with col2:
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=20.0, value=1.0, step=0.1)
        hd_duration = st.number_input("HD Duration (months)", min_value=0, value=0)
        cause = st.selectbox("Cause of Disease", ["HTN", "DM", "GN", "IgA", "FSGS", "PCKD", "Unknown", "etc."])
        hd_type = st.selectbox("HD Type", ["Preemptive", "HD", "CAPD", "HD+PD"])
    
    st.write("### 🏥 Medical History")
    c1, c2 = st.columns(2)
    with c1:
        is_dm = st.checkbox("Diabetes Mellitus (DM)", value=False)
    with c2:
        is_htn = st.checkbox("Hypertension (HTN)", value=False)

    st.write("### 💊 Recent Dosing & TDM")
    st.write("Please enter the most recent day's data.")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        prev_pm = st.number_input("Previous PM Dose (mg)", min_value=0.0, value=2.0, step=0.25)
    with col_d2:
        curr_am = st.number_input("Current AM Dose (mg)", min_value=0.0, value=2.0, step=0.25)
    with col_d3:
        curr_tdm = st.number_input("Current TDM (ng/mL)", min_value=0.0, value=5.0, step=0.1)

    # Hidden assumption: we are simulating 'Day > 4' for stable prediction unless specified
    day_number = st.number_input("Days Post-Transplant", min_value=1, value=30, help="Used for early-stage heuristics")

    submit = st.form_submit_button("Predict Dosage")

if submit:
    # 1. Prepare Static Features
    # Mappings
    cause_map = {"HTN": 0, "DM": 1, "GN": 2, "IgA": 3, "FSGS": 4, "PCKD": 5, "Unknown": 6, "etc.": 7}
    hd_map = {"Preemptive": 0, "HD": 1, "CAPD": 2, "HD+PD": 3}
    
    # Construct tensor: Age, Sex, Bwt, Ht, BMI, Cause, HD_type, HD_duration, DM, HTN
    # Note: Sex: Female=1, Male=0
    sex_val = 1.0 if sex == "Female" else 0.0
    
    static_vals = [
        float(age),
        sex_val,
        float(bwt),
        float(ht),
        float(bmi),
        float(cause_map.get(cause, 7)),
        float(hd_map.get(hd_type, 0)),
        float(hd_duration),
        float(1.0 if is_dm else 0.0),
        float(1.0 if is_htn else 0.0)
    ]
    
    static_tensor = torch.FloatTensor([static_vals])
    
    # 2. PM Prediction
    pm_model, pm_params, pm_err = load_lstm_model('pm', use_static=True, checkpoint_dir=CHECKPOINT_DIR)
    
    if pm_model is None:
        # Fallback to non-static if static not found
         pm_model, pm_params, pm_err = load_lstm_model('pm', use_static=False, checkpoint_dir=CHECKPOINT_DIR)
    
    if pm_err:
        st.error(f"Error loading PM model: {pm_err}")
    else:
        try:
            # Prepare sequence for PM
            # Input: [prev_pm, am, tdm]
            # Since we only have one day of input in this simple UI, we treat it as sequence length 1 (padded)
            # or replicate logic. The training used max_seq_len.
            
            # Create single step sequence
            input_seq = [prev_pm, curr_am, curr_tdm]
            seq_array = np.array([input_seq], dtype=np.float32) # (1, 3)
            
            # Pad
            max_len = pm_params['max_seq_len']
            if len(seq_array) < max_len:
                 padding = np.zeros((max_len - 1, 3), dtype=np.float32)
                 seq_array_padded = np.vstack([seq_array, padding]) # (max_len, 3) - Usually we pad at end? The helper said pad at end.
                 # Wait, existing logic: sequence.append -> pad at END. Yes.
            else:
                 seq_array_padded = seq_array
            
            # Batch dimension
            seq_tensor = torch.FloatTensor(seq_array_padded).unsqueeze(0) # (1, max_len, 3)
            
            # Run Inference
            with torch.no_grad():
                # Provide lengths=1 because we only have 1 valid day
                pm_raw = pm_model(seq_tensor, lengths=torch.LongTensor([1]), static_features=static_tensor if pm_params['use_static'] else None)
                pm_val = pm_raw.item()
                pm_rounded = round_prediction(pm_val)
            
            # 3. AM Prediction
            am_model, am_params, am_err = load_lstm_model('am', use_static=True, checkpoint_dir=CHECKPOINT_DIR)
            if am_model is None:
                am_model, am_params, am_err = load_lstm_model('am', use_static=False, checkpoint_dir=CHECKPOINT_DIR)
                
            if am_err:
                st.error(f"Error loading AM model: {am_err}")
            else:
                # Prepare sequence for AM
                # Input: [prev_pm, am, tdm, predicted_pm]
                input_seq_am = [prev_pm, curr_am, curr_tdm, pm_rounded]
                seq_array_am = np.array([input_seq_am], dtype=np.float32)
                
                max_len_am = am_params['max_seq_len']
                if len(seq_array_am) < max_len_am:
                    padding_am = np.zeros((max_len_am - 1, 4), dtype=np.float32)
                    seq_array_am_padded = np.vstack([seq_array_am, padding_am])
                else:
                    seq_array_am_padded = seq_array_am
                    
                seq_tensor_am = torch.FloatTensor(seq_array_am_padded).unsqueeze(0)
                
                with torch.no_grad():
                    am_raw = am_model(seq_tensor_am, lengths=torch.LongTensor([1]), static_features=static_tensor if am_params['use_static'] else None)
                    am_val = am_raw.item()
                    am_rounded = round_prediction(am_val)

                # Heuristics (Day <= 4)
                if day_number <= 4:
                    if prev_pm <= 1 or curr_am <= 1:
                        pm_rounded += 0.5
                        am_rounded += 0.5
                    if day_number == 1:
                        pm_rounded -= 0.5
                        if pm_rounded < 0: pm_rounded = 0

                # Display Results
                st.success("Prediction Complete")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("Recommended PM Dose", f"{pm_rounded} mg")
                with res_col2:
                    st.metric("Recommended AM Dose (Next Day)", f"{am_rounded} mg")
                    
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.code(str(e)) # Debug info
