import torch
import torch.onnx
import os
import sys
import numpy as np
from timeseries_models import RNNModel
import re

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')
ONNX_DIR = os.path.join(CURRENT_DIR, 'onnx_models')
os.makedirs(ONNX_DIR, exist_ok=True)

device = torch.device('cpu') # Export on CPU

def parse_model_filename(filename):
    """Extract parameters from filename - copied from existing code"""
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
    
    return {
        'use_static': use_static,
        'rnn_type': rnn_type,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers
    }

def convert_model(model_filename):
    print(f"Converting {model_filename}...")
    model_path = os.path.join(CHECKPOINT_DIR, model_filename)
    params = parse_model_filename(model_filename)
    
    # Determine input dimensions
    if model_filename.startswith('pm'):
        input_dim = 3
        model_name_prefix = 'pm'
    else:
        input_dim = 4
        model_name_prefix = 'am'
        
    static_dim = 10 if params['use_static'] else 0
    
    # Initialize Model
    model = RNNModel(
        input_dim=input_dim,
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        use_static=params['use_static'],
        static_dim=static_dim,
        rnn_type=params['rnn_type']
    )
    
    # Load Weights
    # Load to CPU
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create Dummy Inputs
    batch_size = 1
    seq_len = 5 # Arbitrary for tracing
    
    # Input 1: Sequence (batch, seq_len, input_dim)
    dummy_x = torch.randn(batch_size, seq_len, input_dim)
    
    # Input 2: Lengths (batch,)
    dummy_lengths = torch.tensor([seq_len], dtype=torch.long)
    
    # Input 3: Static Features (batch, static_dim) - Optional
    if params['use_static']:
        dummy_static = torch.randn(batch_size, static_dim)
    else:
        dummy_static = None # ONNX export handles None?
        # Actually forward signature is (x, lengths=None, static_features=None)
        # We should provide valid inputs that match the signature we want to use.
        # If model doesn't use static, we can pass None, but for tracing it helps to be explicit.
        # The forward method signature: forward(self, x, lengths=None, static_features=None)
        
    # Export
    output_filename = model_filename.replace('.pth', '.onnx')
    output_path = os.path.join(ONNX_DIR, output_filename)
    
    # Define inputs for export
    # Note: Tuple order must match forward() arguments
    if params['use_static']:
        dummy_inputs = (dummy_x, dummy_lengths, dummy_static)
        input_names = ['x', 'lengths', 'static_features']
        dynamic_axes = {
            'x': {0: 'batch_size', 1: 'seq_len'},
            'lengths': {0: 'batch_size'},
            'static_features': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    else:
        # If static_features is None, we might perform forward with just (x, lengths)
        # But our predict_cli passes None for static_features explicitly if not used.
        # Let's see if we can trace with None.
        dummy_inputs = (dummy_x, dummy_lengths, None)
        input_names = ['x', 'lengths', 'static_features'] # static_features will be optional/null?
        # Tracing with None might be tricky.
        # Better strategy: If !use_static, the model ignores the 3rd arg.
        # We can pass a dummy tensor anyway, or just omit it from the trace signature?
        # Let's pass a dummy tensor of size (1,1) but model won't use it.
        # Actually, looking at code: 
        # if self.use_static and static_features is not None: ... else: combined = rnn_features
        # So if use_static is False, the 3rd arg is ignored.
        # We can pass None.
        
        dynamic_axes = {
            'x': {0: 'batch_size', 1: 'seq_len'},
            'lengths': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

    # For 'static_features', if it's None in trace, it might cause issues if we try to pass it later?
    # Actually, let's just trace with (x, lengths). 
    # But wait, predict_cli calls: model(seq, lengths, static)
    # The signature is fixed.
    
    # Let's handle the non-static case by passing None during export if Torch supports it, 
    # or by wrapping the model to separate signatures?
    # Simplest: Just use the 3-arg signature.
    
    try:
        torch.onnx.export(
            model,
            dummy_inputs,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=input_names,
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        print(f"Successfully converted to {output_path}")
    except Exception as e:
        print(f"Failed to convert {model_filename}: {e}")

def main():
    if not os.path.exists(CHECKPOINT_DIR):
        print("Checkpoint dir not found")
        return

    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
    for f in files:
        convert_model(f)

if __name__ == "__main__":
    main()
