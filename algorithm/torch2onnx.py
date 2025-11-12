import numpy as np
import onnx
import torch
import os
import argparse

# --- Imports from this project ---
from core.model import build_mcu_model
from core.utils.config import configs, load_config_from_file, update_config_from_args
# ---------------------------------

def convert_onnx(config_path, checkpoint_path, output_path, opset=11, simplify=False):
    """
    Converts a .pth checkpoint from the tiny-training project to an ONNX model.
    """
    
    # 1. Load the configuration file used for training
    # This is ESSENTIAL to build the correct model architecture
    print(f"Loading configuration from: {config_path}")
    load_config_from_file(config_path)

    # 2. Build the model structure
    print(f"Building model: {configs.net_config.net_name} with dim {configs.net_config.embedding_dim}")
    net = build_mcu_model()

    # 3. Load the trained weights
    print(f"Loading checkpoint from: {checkpoint_path}")
    weight = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Get the state_dict (weights) from the checkpoint
    if 'state_dict' not in weight:
        print("Error: Checkpoint does not contain a 'state_dict' key.")
        return
    state_dict = weight['state_dict']

    # Handle weights saved from DistributedDataParallel (if 'module.' prefix exists)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k[7:]] = v  # remove 'module.'
        else:
            cleaned_state_dict[k] = v
            
    net.load_state_dict(cleaned_state_dict, strict=True)
    net.eval()
    print("Successfully loaded trained weights into model structure.")

    # 4. Create a dummy input with the correct normalization
    # The training process uses normalization: (img_tensor * 255 - 127.5)
    # The verification.py script uses: img_data - 127.5
    # So the model expects float inputs in the range [-127.5, 127.5]
    
    # Get image size from config
    img_size = configs.data_provider.image_size
    print(f"Creating dummy input with size (1, 3, {img_size}, {img_size})")
    
    # Create a random image [0, 255]
    img_np = np.random.randint(0, 255, size=(1, 3, img_size, img_size)).astype(np.float32)
    
    # Apply the correct normalization
    img_np = (img_np - 127.5)
    dummy_input = torch.from_numpy(img_np)

    # 5. Export to ONNX
    print(f"Exporting to ONNX file: {output_path}")
    torch.onnx.export(
        net, 
        dummy_input, 
        output_path, 
        input_names=["data"], 
        output_names=["embedding"],
        keep_initializers_as_inputs=False, 
        verbose=False, 
        opset_version=opset,
        dynamic_axes={"data": {0: "batch_size"}, "embedding": {0: "batch_size"}} # Add dynamic batch size
    )

    # 6. Load, check, and optionally simplify the ONNX model
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    
    # Make batch size dynamic (already done in export, but good to double-check)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'

    if simplify:
        print("Simplifying ONNX model...")
        try:
            from onnxsim import simplify
            model, check = simplify(model)
            assert check, "Simplified ONNX model could not be validated"
            print("Simplification successful.")
        except ImportError:
            print("Warning: `onnx-simplifier` not found. Skipping simplification.")
            print("Install with: pip install onnx-simplifier")

    # 7. Save the final model
    onnx.save(model, output_path)
    print(f"Final ONNX model saved to: {output_path}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tiny-Training PyTorch to ONNX converter')
    
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the .yaml config file used for training (e.g., algorithm/configs/face_rec_default.yaml)')
    
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to the input .pth checkpoint file (e.g., runs/.../ckpt.best.pth)')
    
    parser.add_argument('--output', type=str, default=None, 
                        help='Path for the output .onnx file. (default: model.onnx in the same dir as input)')
    
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    
    parser.add_argument('--simplify', action='store_true', default=False,
                        help='Run onnx-simplifier on the exported model')
    
    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "model.onnx")

    # Ensure input files exist
    assert os.path.exists(args.config), f"Config file not found: {args.config}"
    assert os.path.exists(args.input), f"Input checkpoint not found: {args.input}"

    convert_onnx(
        config_path=args.config,
        checkpoint_path=args.input,
        output_path=args.output,
        opset=args.opset,
        simplify=args.simplify
    )