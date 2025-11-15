import os
import os.path as osp
import json
import torch
import tvm
from tvm import relay

# --- Import necessary functions from the project ---

# 1. From compilation/convert/pth2ir.py
from compilation.convert.pth2ir import pth_model_to_ir, generated_backward_graph

# 2. From compilation/mod.py
from compilation.mod import mod_save

# 3. From compilation/mcu_ir_gen.py (copy this helper class)
from tvm.relay.expr_functor import ExprMutator
class ExtractMetaConstants(ExprMutator):
    # (Copy the full class definition from compilation/mcu_ir_gen.py)
    def __init__(self):
        super().__init__()
        self.constants = []
    # ... (rest of the class) ...
    def visit_constant(self, const: relay.expr.Constant):
        # ... (rest of the method) ...
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

# 4. --- Import Your Model's Architecture ---
# You MUST provide this file.
# For example: from my_project.models import MyCustomQuantizedModel
from my_model_definition_file import MyCustomQuantizedModel 

# --- Main conversion logic ---
if __name__ == "__main__":
    
    # --- 1. Define Model Parameters ---
    model_name = "my_custom_model"
    path = f"ir_zoos/{model_name}" # Output directory for IRs
    os.makedirs(path, exist_ok=True)
    
    rs = 128  # Set your model's input resolution (e.g., 128)
    num_classes = 10 # Set your model's class count
    int8_bp = False # Set to True if you use int8 backward pass

# --- 2. Instantiate and Load Your .pth Model ---
    print("Loading custom .pth model checkpoint...")
    
    # --- 2a. Instantiate your model's architecture ---
    # You MUST load the same model architecture that you trained.
    # The mcu_ir_gen.py script uses model builders from the project,
    # so you should probably do the same.

    # Example using the project's 'build_mcu_model' function:
    from algorithm.core.model import build_mcu_model
    from algorithm.core.utils.config import configs, load_config_from_file
    
    # Load the config file you used for training (e.g., transfer.yaml)
    # This is necessary so build_mcu_model() knows what to build
    load_config_from_file("algorithm/configs/transfer.yaml") 
    
    # ***IMPORTANT***: Update these to match your trained model
    configs.net_config.net_name = "mcunet-5fps" # e.g., "mcunet-5fps", "mbv2-w0.35", etc.
    configs.data_provider.num_classes = num_classes # Must match 'num_classes' from Step 1
    configs.net_config.mcu_head_type = "quantized"  # Must match your trained model
    
    print(f"Building model architecture for: {configs.net_config.net_name}")
    model = build_mcu_model()
    
    # --- 2b. Load the checkpoint and extract weights ---
    
    # Set the path to your trained .pth file
    pth_file_path = "path/to/your/ckpt.pth" # This is the file saved by BaseTrainer
    
    print(f"Loading checkpoint from: {pth_file_path}")
    checkpoint = torch.load(pth_file_path, map_location='cpu')
    
    # Extract the state_dict from the checkpoint
    if 'state_dict' not in checkpoint:
        raise KeyError(f"Checkpoint file {pth_file_path} does not contain a 'state_dict' key. "
                       "It may not be a file saved by BaseTrainer.")
    state_dict = checkpoint['state_dict']

    # Handle 'module.' prefix if model was saved with DistributedDataParallel
    # This logic is from the resume() method in base_trainer.py
    if not hasattr(model, 'module'):
        # If model is not parallel, but state_dict is, strip the prefix
        if all(k.startswith('module.') for k in state_dict.keys()):
            print("Stripping 'module.' prefix from state_dict keys.")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load the weights into your model architecture
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully from checkpoint.")
    
    # --- 3. Convert PyTorch Model to Forward IR ---
    print("Converting to forward IR...")
    fwd_mod, real_params, scale_params, op_idx = pth_model_to_ir(
        model, 
        input_res=[1, 3, rs, rs], 
        num_classes=num_classes
    )
    
    # Save scale.json (used by mcu_ir_gen.py)
    with open(f"{path}/scale.json", "w") as fp:
        json.dump(scale_params, fp, indent=2)

    # Save forward IR
    fshape_str = "x".join([str(_) for _ in [1, 3, rs, rs]])
    mod_save(fwd_mod, params=real_params, path=path, mod_name=f"fwd-{fshape_str}.ir")
    print(f"Forward IR saved to {path}/fwd-{fshape_str}.ir")

    # --- 4. Define Sparse Update Config ---
    # As you mentioned it's the same, copy the dict from
    # compilation/mcu_ir_gen.py that matches your model.
    # This example uses the 'proxyless' config.
    sparse_update_config = {
        "49kb": {
            'enable_backward_config': 1, 'n_bias_update': 21, 'n_weight_update': 0, 'weight_update_ratio': [0.25, 1, 0, 1, 0, 0.125, 0.25, 0.25], 'manual_weight_idx': [39, 42, 44, 45, 50, 51, 54, 57], 'weight_select_criteria': 'magnitude+', 'pw1_weight_only': 0
        },
        # ... (add other configs like "74kb", "98kb", etc.)
    }
    print("Generating backward graphs with sparse update configs...")

    # --- 5. Generate Backward IRs ---
    for mem, cfg in sparse_update_config.items():
        print(f"Generating graph for {mem} config...")
        bwd_mod, bwd_names, sparse_meta_info = generated_backward_graph(
            fwd_mod, 
            op_idx, 
            method="sparse_bp", 
            sparse_bp_config=cfg, 
            int8_bp=int8_bp
        )
        
        meta_info = {
            "output_info" : bwd_names,
            "sparse_update_info": sparse_meta_info,
        }

        # Extract constants (same as in mcu_ir_gen.py)
        consts = extract_const_from_mod(bwd_mod)
        
        # Save the sparse backward IR and its metadata
        ir_name = f"sparse_bp-{mem}-{fshape_str}.ir"
        meta_name = f"sparse_bp-{mem}-{fshape_str}.meta"
        
        mod_save(
            bwd_mod,
            None,
            path=f"{path}",
            mod_name=ir_name,
            meta=consts
        )
        with open(osp.join(path, meta_name), "w") as fp:
            json.dump(meta_info, fp, indent=2)
        
        print(f"Saved {ir_name} and {meta_name}")

    print("All IR files generated.")