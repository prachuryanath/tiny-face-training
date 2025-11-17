import os
import sys
import torch
import torch.nn as nn

# --- Imports from your project ---
# Ensure this script is placed in the 'algorithm/' directory
# so that these imports work correctly.
try:
    from core.model import build_mcu_model
    from core.utils.config import configs, load_config_from_file
except ImportError:
    print("Error: Could not import project modules.")
    print("Please make sure you run this script from the 'algorithm/' directory,")
    print("or that the 'algorithm/' directory is in your PYTHONPATH.")
    sys.exit(1)

# --- 1. CONFIGURATION ---
# PLEASE EDIT THESE VALUES to match your trained model
# ---------------------------------------------------

# Path to the checkpoint .pth file saved by base_trainer.py
CHECKPOINT_PATH = "ckpt.best.pth" 

# Path to the .yaml config file you used for training
# This is needed to build the model architecture
CONFIG_FILE_PATH = "configs/config.yaml" # e.g., 'algorithm/configs/vww.yaml'

# Model parameters from your config
NET_NAME = "mcunet-5fps"  # e.g., "mcunet-5fps", "mbv2-w0.35"
NUM_CLASSES = 100          # e.g., 102 for flowers, 2 for vww
MCU_HEAD_TYPE = "fp" # e.g., "quantized" or "fp"
INPUT_SHAPE = (1, 3, 80, 80) # (batch, channels, height, width)

# ---------------------------------------------------
# --- 2. HELPER FUNCTION ---
# ---------------------------------------------------

def build_model_from_config():
    """
    Uses the project's 'build_mcu_model' function to
    create the correct model architecture.
    """
    # Load the base config file
    try:
        load_config_from_file(CONFIG_FILE_PATH)
        print("Config file found")
    except FileNotFoundError:
        print(f"Error: Config file not found at {CONFIG_FILE_PATH}")
        print("Please correct the CONFIG_FILE_PATH variable in this script.")
        sys.exit(1)
        
    # Set the specific parameters for your model
    configs.net_config.net_name = NET_NAME
    configs.data_provider.num_classes = NUM_CLASSES
    configs.net_config.mcu_head_type = MCU_HEAD_TYPE
    
    print(f"Building model architecture for: {NET_NAME} with {NUM_CLASSES} classes...")
    model = build_mcu_model()
    return model

# ---------------------------------------------------
# --- 3. TEST SCRIPT ---
# ---------------------------------------------------

def main():
    print("--- Starting Model Load Check ---")

    # --- Test 1: Build Fresh Model (Baseline) ---
    print("\n[Test 1: Build Fresh Model]")
    try:
        fresh_model = build_model_from_config()
        fresh_model.eval()
        print("  -> SUCCESS: Fresh model built successfully.")
    except Exception as e:
        print(f"  -> FAILED: Could not build fresh model. Error: {e!r}") # Changed 'e' to 'e!r'
        sys.exit(1)

    # --- Test 2: Build and Load Checkpoint Model ---
    print("\n[Test 2: Load Checkpoint]")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"  -> FAILED: Checkpoint file not found at: {CHECKPOINT_PATH}")
        print("     Please correct the CHECKPOINT_PATH variable in this script.")
        sys.exit(1)
        
    try:
        loaded_model = build_model_from_config()
        
        print(f"  Loading checkpoint from: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

        # As seen in base_trainer.py, weights are under the 'state_dict' key
        if 'state_dict' not in checkpoint:
            print("  -> FAILED: 'state_dict' key not found in checkpoint file.")
            sys.exit(1)
            
        state_dict = checkpoint['state_dict']

        # Handle 'module.' prefix if model was saved with DistributedDataParallel
        if not hasattr(loaded_model, 'module') and all(k.startswith('module.') for k in state_dict.keys()):
            print("  Stripping 'module.' prefix from state_dict keys...")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Load the weights
        # --- CHANGE IS HERE ---
        # We use strict=False to ignore the 'keep_mask' keys,
        # which are part of the training setup but not the base model.
        print("  Loading state_dict with strict=False to ignore 'keep_mask' buffers...")
        load_result = loaded_model.load_state_dict(state_dict, strict=False)
        
        # Report what was ignored
        if load_result.unexpected_keys:
            print(f"    -> Ignored {len(load_result.unexpected_keys)} unexpected keys (as expected):")
            print(f"       {load_result.unexpected_keys[:5]}...") # Print first 5
        if load_result.missing_keys:
             print(f"    -> WARNING: Missing {len(load_result.missing_keys)} keys: {load_result.missing_keys}")

        loaded_model.eval()
        print("  -> SUCCESS: Checkpoint 'state_dict' loaded into model architecture.")
    
    except Exception as e:
        print(f"  -> FAILED: Error during model load. This often means the model architecture")
        print(f"     (from your config) does not match the weights in the checkpoint.")
        print(f"     Error details: {e!r}") # Changed 'e' to 'e!r'
        sys.exit(1)

    # --- Test 3: Forward Pass (Sanity Check) ---
    print("\n[Test 3: Forward Pass Sanity Check]")
    try:
        print(f"  Creating dummy input tensor of shape: {INPUT_SHAPE}")
        dummy_input = torch.randn(*INPUT_SHAPE)
        
        print("  Running forward pass on FRESH model...")
        with torch.no_grad():
            output_fresh = fresh_model(dummy_input)
        print(f"    -> Fresh model output shape: {output_fresh.shape}")
        
        print("  Running forward pass on LOADED model...")
        with torch.no_grad():
            output_loaded = loaded_model(dummy_input)
        print(f"    -> Loaded model output shape: {output_loaded.shape}")
        
        assert output_fresh.shape == output_loaded.shape, "Output shapes do not match!"
        assert output_fresh.shape[-1] == NUM_CLASSES, f"Output shape ({output_fresh.shape[-1]}) does not match NUM_CLASSES ({NUM_CLASSES})"
        print("  -> SUCCESS: Forward pass completed on both models. Output shapes are correct.")
        
    except Exception as e:
        print(f"  -> FAILED: Error during forward pass. Error: {e!r}") # Changed 'e' to 'e!r'
        sys.exit(1)

    # --- Test 4: Compare Outputs (Verification) ---
    print("\n[Test 4: Output Verification]")
    
    # Compare the outputs of the fresh vs. loaded model
    diff = torch.abs(output_fresh - output_loaded).sum()
    
    if diff == 0.0:
        print(f"  -> WARNING: Outputs from fresh and loaded models are IDENTICAL (diff: {diff}).")
        print("     This is highly unlikely if the checkpoint was trained.")
        print("     Are you sure the CHECKPOINT_PATH is correct and not an initial state?")
    else:
        print(f"  -> SUCCESS: Outputs are different (sum of abs diff: {diff}).")
        print("     This indicates the loaded weights are active and different from random initialization.")

    print("\n--- Model Load Check Complete. All tests passed! ---")

if __name__ == "__main__":
    main()