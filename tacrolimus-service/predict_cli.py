import argparse
import json
import torch
import os
import sys
import numpy as np
import re
from timeseries_models import RNNModel

# --- Configuration & Helpers (Copied from app.py) ---

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def load_lstm_model(model_type, use_static, checkpoint_dir):
    """RNN Model Loader"""
    try:
        all_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        model_files = [f for f in all_files 
                      if f.startswith(f'{model_type}_lstm') or f.startswith(f'{model_type}_gru')]
        
        if use_static:
            model_files = [f for f in model_files if 'static' in f]
        else:
            model_files = [f for f in model_files if 'static' not in f]
        
        if not model_files:
            return None, None, f"{model_type.upper()} {'static' if use_static else 'non-static'} model file not found."
        
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
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        return model, params, None
        
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"

def round_prediction(value):
    if value < 0:
        return 0
    else:
        return round(value * 2) / 2

def prepare_sequence_for_prediction(patient_data, day_index, model_type, max_seq_len, pm_prediction=None):
    """Prepare sequence data matching app.py logic exactly"""
    sequence = []
    
    # Range 1 to day_index (exclusive)
    for day in range(1, day_index):
        day_str = str(day)
        if day_str not in patient_data: 
             continue
        
        day_data = patient_data[day_str]
        
        # Parse inputs carefully, handling nulls/empty strings
        prev_pm = float(day_data.get('prev_pm', 0) or 0)
        am = float(day_data.get('am', 0) or 0)
        tdm = float(day_data.get('tdm', 0) or 0)
        
        if am > 0 or tdm > 0:
            if model_type == 'pm':
                sequence.append([prev_pm, am, tdm])
            else:
                pm_dose = float(day_data.get('predicted_pm', 0) or 0)
                # Note: app.py uses '당일 오후 FK용량' which is saved from previous predictions.
                # In our React app, we store this in 'predicted_pm'.
                sequence.append([prev_pm, am, tdm, pm_dose])
    
    # Current day
    current_day_str = str(day_index)
    if current_day_str in patient_data:
        day_data = patient_data[current_day_str]
        prev_pm = float(day_data.get('prev_pm', 0) or 0)
        am = float(day_data.get('am', 0) or 0)
        tdm = float(day_data.get('tdm', 0) or 0)
        
        if model_type == 'pm':
            sequence.append([prev_pm, am, tdm])
        else:
             # For AM model, we need the today_pm dose.
             # If passed (e.g. from just-calculated prediction), use it.
             # Otherwise try to find in data (React app stores it in 'predicted_pm')
             if pm_prediction is not None:
                 today_dose_pm = pm_prediction 
             else:
                 today_dose_pm = float(day_data.get('predicted_pm', 0) or 0)
                 
             sequence.append([prev_pm, am, tdm, today_dose_pm])
    
    if not sequence:
        return None, None

    sequence_array = np.array(sequence, dtype=np.float32)
    seq_len = len(sequence_array)
    
    if seq_len < max_seq_len:
        padding = np.zeros((max_seq_len - seq_len, sequence_array.shape[1]), dtype=np.float32)
        sequence_array = np.vstack([sequence_array, padding])
    elif seq_len > max_seq_len:
        # app.py logic actually pads to max_seq_len if smaller, but if larger?
        # The parser regex implies models are trained with fixed max_seq_len (e.g. 10 or 23).
        # We should probably truncate the *start* if it's too long?
        # But app.py prepares valid sequence. Let's assume typical sequence is short (<8 days).
        # If > max_seq_len, we take the *last* max_seq_len items
        sequence_array = sequence_array[-max_seq_len:]
        seq_len = max_seq_len
    
    sequence_tensor = torch.FloatTensor(sequence_array).unsqueeze(0).to(device)
    seq_length = torch.LongTensor([seq_len]).cpu()
    
    return sequence_tensor, seq_length


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='JSON input')
    args = parser.parse_args()
    
    try:
        data = json.loads(args.input)
        
        patient_data = data.get('patient_data', {})
        target_day = int(data.get('target_day', 1))
        static_features_dict = data.get('static_features', {})
        
        # 1. Determine Static Features usage
        # app.py: use_static = len(static_features_dict) == 10
        # We'll trust the dictionary passed.
        # Filter keys that have valid values
        valid_static = {k: v for k, v in static_features_dict.items() if v not in [None, ""]}
        use_static = len(valid_static) == 10
        
        # Prepare Static Tensor
        static_feature_tensor = None
        if use_static:
            try:
                vals = [
                    float(valid_static.get('AGE', 0)),
                    float(1.0 if valid_static.get('Sex') == 'Female' else 0.0), # Helper handles string conversion
                    float(valid_static.get('Bwt', 0)),
                    float(valid_static.get('Ht', 0)),
                    float(valid_static.get('BMI', 0)),
                    float(valid_static.get('Cause', 0)), # Assuming mapped value sent or need mapping? 
                    # React app sends strings/values. We might need mapping if strings sent.
                    # Looking at page.tsx, select values are: "HTN", "DM"...
                    # We need to map these strings to float codes as per app.py logic
                    # app.py mapping:
                    # Cause: HTN=0, DM=1, GN=2, IgA=3, FSGS=4, PCKD=5, Unknown=6, etc=7
                    # HD_type: Preemptive=0, HD=1, CAPD=2, HD+PD=3
                    # DM/HTN: No=0, Yes=1
                    
                    # NOTE: To be safe, let's map here.
                    0.0, # Placeholder Cause
                    0.0, # Placeholder HD_type
                    float(valid_static.get('HD_duration', 0)),
                    0.0, # Placeholder DM
                    0.0  # Placeholder HTN
                ]
                
                # Mappings
                cause_map = {"HTN": 0, "DM": 1, "GN": 2, "IgA": 3, "FSGS": 4, "PCKD": 5, "Unknown": 6, "etc.": 7}
                vals[5] = float(cause_map.get(valid_static.get('Cause'), 0))
                
                hd_map = {"Preemptive": 0, "HD": 1, "CAPD": 2, "HD+PD": 3}
                vals[6] = float(hd_map.get(valid_static.get('HD_type'), 0))

                bool_map = {"No": 0, "Yes": 1}
                vals[8] = float(bool_map.get(valid_static.get('DM'), 0))
                vals[9] = float(bool_map.get(valid_static.get('HTN'), 0))

                static_feature_tensor = torch.FloatTensor([vals]).to(device)
            except Exception as e:
                # Fallback to non-static if parsing fails
                use_static = False
                static_feature_tensor = None

        # 2. Load PM Model
        pm_model, pm_params, err = load_lstm_model('pm', use_static, CHECKPOINT_DIR)
        if err: raise Exception(err)
        
        # 3. Predict PM
        pm_seq, pm_len = prepare_sequence_for_prediction(patient_data, target_day, 'pm', pm_params['max_seq_len'])
        if pm_seq is None: raise Exception("Insufficient data for PM prediction")

        with torch.no_grad():
            pm_raw = pm_model(pm_seq, lengths=pm_len, static_features=static_feature_tensor)
            pm_val = pm_raw.cpu().item()
        
        pm_rounded = round_prediction(pm_val)
        
        # 4. Load AM Model
        am_model, am_params, err = load_lstm_model('am', use_static, CHECKPOINT_DIR)
        if err: raise Exception(err)
        
        # 5. Predict AM (using the just-predicted PM value)
        am_seq, am_len = prepare_sequence_for_prediction(patient_data, target_day, 'am', am_params['max_seq_len'], pm_prediction=pm_rounded)
        if am_seq is None: raise Exception("Insufficient data for AM prediction")
        
        with torch.no_grad():
            am_raw = am_model(am_seq, lengths=am_len, static_features=static_feature_tensor)
            am_val = am_raw.cpu().item()
            
        am_rounded = round_prediction(am_val)
        
        # 6. Apply Heuristics
        # Logic: Low dose correction (0~4 day)
        # Check current inputs
        curr_prev_pm = float(patient_data.get(str(target_day), {}).get('prev_pm', 0))
        curr_am = float(patient_data.get(str(target_day), {}).get('am', 0))
        
        if target_day <= 4:
            if curr_prev_pm <= 1 or curr_am <= 1:
                pm_rounded += 0.5
                am_rounded += 0.5
        
        if target_day == 1:
            pm_rounded -= 0.5
            
        # 7. Output Result
        print(json.dumps({
            "status": "success",
            "predicted_pm": pm_rounded,
            "predicted_am": am_rounded
        }))

    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))

if __name__ == "__main__":
    main()
