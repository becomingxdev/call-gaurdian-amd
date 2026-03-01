import onnxruntime as ort
import numpy as np
import os

def verify_model():
    onnx_path = "models/deepfake_detector.onnx"
    if not os.path.exists(onnx_path):
        print(f"Error: {onnx_path} not found.")
        return

    print(f"Loading model: {onnx_path}")
    try:
        # Load with DirectML
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get metadata
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        
        print("\n--- Model Metadata ---")
        print(f"Model Input Name: {input_meta.name}")
        print(f"Model Input Shape: {input_meta.shape}")
        print(f"Model Output Name: {output_meta.name}")
        print(f"Model Output Shape: {output_meta.shape}")
        
        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"File Size: {file_size_mb:.2f} MB")
        
        print("\n--- Execution Status ---")
        print(f"Inference session loaded successfully with providers: {session.get_providers()}")
        
        # Test a dummy inference
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {input_meta.name: dummy_input})
        print(f"Dummy inference successful. Output prob: {outputs[0][0][0]:.4f}")

    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    verify_model()
