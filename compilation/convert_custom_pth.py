import os
import os.path as osp
import json
import torch
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator
import sys

# --- Project-specific Imports ---

# Add algorithm directory to path to find core modules
project_root = os.path.dirname(os.path.abspath(__file__))
algorithm_path = os.path.join(project_root, '..', 'algorithm')
sys.path.append(algorithm_path)

try:
    # Conversion tools
    from compilation.convert.pth2ir import pth_model_to_ir, generated_backward_graph
    from compilation.mod import mod_save

    # Model building tools
    from core.model import build_mcu_model
    from core.utils.config import configs, load_config_from_file
except ImportError as e:
    print(f"Error: Failed to import project modules: {e}")
    print("Please make sure you place this script in the 'compilation/' directory,")
    print("and that the 'algorithm/' directory exists at the same level as 'compilation/'.")
    sys.exit(1)

# ---------------------------------------------------
# --- 1. CONFIGURATION ---
# PLEASE VERIFY THESE VALUES
# ---------------------------------------------------

# --- Values from your config file ---
NET_NAME = "mcunet-5fps"
NUM_CLASSES = 100
MCU_HEAD_TYPE = "fp"
IMAGE_SIZE = 80
INT8_BP = False # From your config (quantize_gradient: 0)

# --- Paths ---
# Path to the .yaml config file you used for training
# This is needed to build the base model architecture
CONFIG_FILE_PATH = os.path.join(algorithm_path, "configs", "transfer.yaml")

# Path to the FINETUNED checkpoint .pth file
CHECKPOINT_PATH = "runs/face-100/mcunet-5fps/80_embed/sparse_49kb/sgd_qas_nomom/checkpoint/ckpt.best.pth" # <<<--- VERIFY THIS PATH

# Where to save the generated IR files
OUTPUT_DIR = f"ir_zoos/{NET_NAME}_finetuned_face"

# ---------------------------------------------------

# Helper class copied from compilation/mcu_ir_gen.py
# This is needed to handle TVM MetaConstants
class ExtractMetaConstants(ExprMutator):
    def __init__(self):
        super().__init__()
        self.constants = []

    def visit_constant(self, const: relay.expr.Constant):
        new_const = relay.const(const.data.numpy())
        np_data = const.data.numpy()
        if np_data.size == 1:
            value = np_data.item()
            new_const = relay.const(value, dtype=str(np_data.dtype))
        if "meta" in str(const):
            self.constants.append(np_data)
        return new_const

    def extract_constants(self, func):
        expr = self.visit(func)
        return expr, self.constants

def extract_const_from_mod(mod):
    func = mod['main']
    new_func, consts = ExtractMetaConstants().extract_constants(func)
    return consts

def load_finetuned_model():
    """
    Builds the base model architecture and loads the finetuned .pth checkpoint.
    """
    print(f"--- Loading Finetuned Model ---")
    
    # 1. Load base config file
    try:
        load_config_from_file(CONFIG_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Base config file not found at {CONFIG_FILE_PATH}")
        sys.exit(1)
        
    # 2. Set specific parameters for your finetuned model
    configs.net_config.net_name = NET_NAME
    configs.data_provider.num_classes = NUM_CLASSES
    configs.net_config.mcu_head_type = MCU_HEAD_TYPE
    configs.data_provider.image_size = IMAGE_SIZE
    
    print(f"Building base model architecture for: {NET_NAME}...")
    try:
        model = build_mcu_model()
    except Exception as e:
        print(f"Error building model architecture: {e!r}")
        print("Please ensure 'algorithm/core/model/model_entry.py' has the correct path to the .pkl files.")
        sys.exit(1)
    
    # 3. Load the checkpoint .pth file
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at: {CHECKPOINT_PATH}")
        sys.exit(1)
        
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    
    if 'state_dict' not in checkpoint:
        raise KeyError("Checkpoint file does not contain a 'state_dict' key.")
        
    state_dict = checkpoint['state_dict']

    # 4. Handle 'module.' prefix if model was saved with DistributedDataParallel
    if not hasattr(model, 'module'):
        if all(k.startswith('module.') for k in state_dict.keys()):
            print("  Stripping 'module.' prefix from state_dict keys.")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 5. Load weights, ignoring 'keep_mask' keys
    print("  Loading state_dict with strict=False...")
    load_result = model.load_state_dict(state_dict, strict=False)
    
    if load_result.unexpected_keys:
        print(f"  -> Ignored {len(load_result.unexpected_keys)} unexpected keys (e.g., keep_mask).")
    if load_result.missing_keys:
         print(f"  -> WARNING: Missing {len(load_result.missing_keys)} keys: {load_result.missing_keys}")

    model.eval()
    print("--- Model Loaded Successfully ---")
    return model

def main():
    print(f"--- Starting IR Conversion ---")
    print(f"Output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 1. Load Model ---
    model = load_finetuned_model()
    
    # --- 2. Convert to Forward IR ---
    print("\n--- Phase 1: Converting to Forward IR ---")
    input_res = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    
    fwd_mod, real_params, scale_params, op_idx = pth_model_to_ir(
        model, 
        input_res=input_res, 
        num_classes=NUM_CLASSES
    )
    
    # Save scale.json
    with open(f"{OUTPUT_DIR}/scale.json", "w") as fp:
        json.dump(scale_params, fp, indent=2)
    print("  Saved scale.json")

    # Save forward IR
    fshape_str = "x".join([str(_) for _ in input_res])
    fwd_ir_name = f"fwd-{fshape_str}.ir"
    mod_save(fwd_mod, params=real_params, path=OUTPUT_DIR, mod_name=fwd_ir_name)
    print(f"  Saved forward IR: {fwd_ir_name}")

    # --- 3. Define Sparse Update Config ---
    # This config is for 'mcunet-5fps' copied from mcu_ir_gen.py
    # Your config file matched the "49kb" sparse config
    sparse_update_config = {
        "49kb": {
            "enable_backward_config": 1, "n_bias_update": 20, "n_weight_update": 0, "weight_update_ratio": [0, 0.25, 0.5, 0.5, 0, 0], "manual_weight_idx": [23, 24, 27, 30, 33, 39], "weight_select_criteria": "magnitude+", "pw1_weight_only": 0,
        },
        "74kb": {
            'enable_backward_config': 1, 'n_bias_update': 21, 'n_weight_update': 0, 'weight_update_ratio': [1, 0, 1, 0, 0.5, 1], 'manual_weight_idx': [21, 23, 24, 26, 27, 30], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        },
        "99kb": {
            'enable_backward_config': 1, 'n_bias_update': 22, 'n_weight_update': 0, 'weight_update_ratio': [1, 1, 1, 1, 0.125, 0.25], 'manual_weight_idx': [21, 24, 27, 30, 36, 39], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        },
        "124kb": {
            'enable_backward_config': 1, 'n_bias_update': 24, 'n_weight_update': 0, 'weight_update_ratio': [0.25, 1, 1, 1, 0.5, 0.5], 'manual_weight_idx': [21, 24, 27, 30, 33, 39], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        },
        "148kb": {
            'enable_backward_config': 1, 'n_bias_update': 23, 'n_weight_update': 0, 'weight_update_ratio': [1, 1, 1, 1, 1, 0.5], 'manual_weight_idx': [21, 24, 27, 30, 36, 39], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        }
    }
    
    print("\n--- Phase 2: Generating Sparse Backward IRs ---")

    # --- 4. Generate Backward IRs ---
    for mem, cfg in sparse_update_config.items():
        print(f"  Generating graph for {mem} config...")
        bwd_mod, bwd_names, sparse_meta_info = generated_backward_graph(
            fwd_mod, 
            op_idx, 
            method="sparse_bp", 
            sparse_bp_config=cfg, 
            int8_bp=INT8_BP
        )
        
        meta_info = {
            "output_info" : bwd_names,
            "sparse_update_info": sparse_meta_info,
        }

        # Extract constants
        consts = extract_const_from_mod(bwd_mod)
        
        # Save the sparse backward IR and its metadata
        ir_name = f"sparse_bp-{mem}-{fshape_str}.ir"
        meta_name = f"sparse_bp-{mem}-{fshape_str}.meta"
        
        mod_save(
            bwd_mod,
            None,
            path=f"{OUTPUT_DIR}",
            mod_name=ir_name,
            meta=consts
        )
        with open(osp.join(OUTPUT_DIR, meta_name), "w") as fp:
            json.dump(meta_info, fp, indent=2)
        
        print(f"    -> Saved {ir_name} and {meta_name}")

    print("\n--- All IR files generated successfully! ---")
    print("\nNext step: Run `compilation/ir2json.py` on your new .ir files.")
    print(f"Example: python compilation/ir2json.py {OUTPUT_DIR}/sparse_bp-49kb-{fshape_str}.ir")

if __name__ == "__main__":
    main()