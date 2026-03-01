import torch
import torch.nn as nn
from torchvision.models import resnet18
import os

def export_model():
    print("Preparing ResNet18 for export...")
    
    # 1. Initialize ResNet18
    # In a real scenario, we would load pre-trained weights here.
    # For this prototype/demo, we use the architecture and export it.
    model = resnet18(weights=None)
    
    # 2. Modify final layer for binary classification (Real vs Fake)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    
    model.eval()
    
    # 3. Create dummy input
    # Shape: (Batch, Channels, Height, Width) -> (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 4. Define paths
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/deepfake_detector.onnx"
    
    # 5. Export to ONNX
    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete.")

if __name__ == "__main__":
    export_model()
