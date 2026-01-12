import argparse
import json
import torch
import os
import sys
import numpy as np
from timeseries_models import RNNModel
import re

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_model_filename(filename):
    """Extract parameters from filename - copied from app.py"""
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

def load_model(model_type, use_static):
    """Load model - adapted from app.py"""
    all_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
    model_files = [f for f in all_files 
                  if f.startswith(f'{model_type}_lstm') or f.startswith(f'{model_type}_gru')]
    
    if use_static:
        model_files = [f for f in model_files if 'static' in f]
    else:
        model_files = [f for f in model_files if 'static' not in f]
    
    if not model_files:
        raise FileNotFoundError(f"Could not find model file for {model_type} (static={use_static})")
    
    model_file = model_files[0]
    model_path = os.path.join(CHECKPOINT_DIR, model_file)
    
    params = parse_model_filename(model_file)
    
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
        rnn_type=params['rnn_type']
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, params

def round_prediction(value):
    if value < 0:
        return 0
    else:
        return round(value * 2) / 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='JSON string input')
    args = parser.parse_args()
    
    try:
        data = json.loads(args.input)
        
        # Extract inputs
        # We expect data to have:
        # patient_data: list of day objects
        # target_day: int
        # static_features: dict (optional)
        
        patient_data = data.get('patient_data', {}) # Map of "1": {...}, "2": {...}
        target_day = int(data.get('target_day', 1))
        static_features_dict = data.get('static_features', {})
        
        # Check if static features are complete (10 items)
        # In the app, they check specific dictionary keys. We'll simplify:
        use_static = len(static_features_dict) == 10
        
        # Prepare Static Tensor
        static_tensor = None
        if use_static:
            # Order: AGE, Sex, Bwt, Ht, BMI, Cause, HD_type, HD_duration, DM, HTN
            vals = [
                float(static_features_dict.get('AGE', 0)),
                float(static_features_dict.get('Sex', 0)),
                float(static_features_dict.get('Bwt', 0)),
                float(static_features_dict.get('Ht', 0)),
                float(static_features_dict.get('BMI', 0)),
                float(static_features_dict.get('Cause', 0)),
                float(static_features_dict.get('HD_type', 0)),
                float(static_features_dict.get('HD_duration', 0)),
                float(static_features_dict.get('DM', 0)),
                float(static_features_dict.get('HTN', 0))
            ]
            static_tensor = torch.FloatTensor([vals]).to(device)

        # Load PM Model
        pm_model, pm_params = load_model('pm', use_static)
        
        # Prepare PM Sequence
        # Sequence logic: from day 1 to day (target_day)
        # Input: prev_pm, am, tdm.
        sequence = []
        
        # Using 1-based index from inputs
        for d in range(1, target_day + 1):
             day_key = str(d)
             if day_key in patient_data:
                 dd = patient_data[day_key]
                 prev_pm = float(dd.get('prev_pm', 0))
                 am = float(dd.get('am', 0))
                 tdm = float(dd.get('tdm', 0))
                 
                 # Logic from app.py prepare_sequence_for_prediction
                 # app.py loops range(1, day_index) then adds current day
                 # effectively it builds a sequence of all available data points up to now
                 
                 # PM Input: [prev_pm, am, tdm]
                 sequence.append([prev_pm, am, tdm])
        
        # Pad and Tensorize
        max_seq_len = pm_params['max_seq_len']
        if not sequence:
             print(json.dumps({"error": "No data sequence"}))
             return

        seq_array = np.array(sequence, dtype=np.float32)
        seq_len = len(seq_array)
        
        if seq_len < max_seq_len:
            padding = np.zeros((max_seq_len - seq_len, 3), dtype=np.float32) # 3 for PM
            seq_array = np.vstack([seq_array, padding])
        elif seq_len > max_seq_len:
             # Truncate to last max_seq_len? The original code doesn't seem to truncate, 
             # assumes day index won't exceed significantly or logic handles it.
             # Original code: loops to day_index.
             seq_array = seq_array[-max_seq_len:]
             seq_len = max_seq_len

        seq_tensor = torch.FloatTensor(seq_array).unsqueeze(0).to(device)
        seq_length_tensor = torch.LongTensor([seq_len]).cpu()
        
        # Predict PM
        with torch.no_grad():
            pm_pred = pm_model(seq_tensor, lengths=seq_length_tensor, static_features=static_tensor)
            pm_pred_val = pm_pred.cpu().item()
            
        pm_pred_rounded = round_prediction(pm_pred_val)
        
        # Apply heuristics
        # Logic: 1st day predict PM -= 0.5
        if target_day == 1:
            pm_pred_rounded -= 0.5
            
        # Logic: Low dose correction (0~4 day)
        current_prev_pm = float(patient_data.get(str(target_day), {}).get('prev_pm', 0))
        current_am = float(patient_data.get(str(target_day), {}).get('am', 0))
        
        low_dose_corr = 0.0
        if target_day <= 4:
            if current_prev_pm <= 1 or current_am <= 1:
                low_dose_corr = 0.5

        final_pm = pm_pred_rounded + low_dose_corr
        
        # Predict AM (Next Day)
        # AM Model input dim 4: [prev_pm, am, tdm, pm_dose]
        # We need to rebuild sequence with 4 features.
        am_model, am_params = load_model('am', use_static)
        
        am_sequence = []
        for d in range(1, target_day + 1):
            day_key = str(d)
            if day_key in patient_data:
                 dd = patient_data[day_key]
                 prev_pm = float(dd.get('prev_pm', 0))
                 am = float(dd.get('am', 0))
                 tdm = float(dd.get('tdm', 0))
                 
                 # PM Dose: For past days, we might not have it in this simple input structure?
                 # Wait, app.py uses "당일 오후 FK용량".
                 # In our DayData struct, we might not be storing 'actual' PM dose for past days if it wasn't predicted/entered.
                 # But app.py: "day_data.get('당일 오후 FK용량', 0)"
                 # For the TARGET day, we use the `pm_pred_rounded` (before correction? or after?)
                 # app.py: "today_dose_pm = pm_prediction if pm_prediction is not None else ..."
                 # It passes pm_prediction (unrounded? No, it rounds it immediately after inference).
                 
                 # We need `today_dose_pm` for past days. If missing, 0.
                 # In this CLI, we rely on passed data.
                 # Let's assume passed data has 'pm' property if available, or we use 0.
                 
                 pm_dose = float(dd.get('pm', 0))
                 if d == target_day:
                     pm_dose = final_pm # Use the predicted PM for current step
                
                 am_sequence.append([prev_pm, am, tdm, pm_dose])
        
        # Pad AM
        max_seq_len_am = am_params['max_seq_len']
        if not am_sequence:
             # Should not happen
             am_seq_array = np.zeros((1, 4), dtype=np.float32)
             am_seq_len = 1
        else:
            am_seq_array = np.array(am_sequence, dtype=np.float32)
            am_seq_len = len(am_seq_array)
            
            if am_seq_len < max_seq_len_am:
                padding = np.zeros((max_seq_len_am - am_seq_len, 4), dtype=np.float32)
                am_seq_array = np.vstack([am_seq_array, padding])
            elif am_seq_len > max_seq_len_am:
                am_seq_array = am_seq_array[-max_seq_len_am:]
                am_seq_len = max_seq_len_am

        am_seq_tensor = torch.FloatTensor(am_seq_array).unsqueeze(0).to(device)
        am_seq_length_tensor = torch.LongTensor([am_seq_len]).cpu()
        
        with torch.no_grad():
            am_pred = am_model(am_seq_tensor, lengths=am_seq_length_tensor, static_features=static_tensor)
            am_pred_val = am_pred.cpu().item()
            
        final_am = round_prediction(am_pred_val) + low_dose_corr
        
        result = {
            "predicted_pm": final_pm,
            "predicted_am": final_am,
            "status": "success"
        }
        
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e), "status": "error"}))

if __name__ == "__main__":
    main()
