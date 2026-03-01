import numpy as np
import onnxruntime as ort
import os


class DeepfakeDetector:
    """
    Call-Guardian Deepfake Voice Detector
    ======================================
    Loads a binary deepfake detection ONNX model and runs inference
    using AMD DirectML (GPU) as the primary execution provider.

    Input:  Mel-spectrogram image as np.ndarray with shape (224, 224, 3)
    Output: Confidence score from 0.0 to 100.0
            0   = Highly likely FAKE
            100 = Highly likely REAL
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "deepfake_detector.onnx")
    PROVIDERS = ["DmlExecutionProvider", "CPUExecutionProvider"]

    def __init__(self):
        """
        Initialise the ONNX inference session with AMD DirectML provider.
        Stores input and output node names for use during inference.
        """
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at: {self.MODEL_PATH}\n"
                "Please ensure 'models/deepfake_detector.onnx' exists in the project root."
            )

        print(f"[DeepfakeDetector] Loading model from: {self.MODEL_PATH}")
        self.session = ort.InferenceSession(self.MODEL_PATH, providers=self.PROVIDERS)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        active_providers = self.session.get_providers()
        print(f"[DeepfakeDetector] Active execution providers: {active_providers}")
        print(f"[DeepfakeDetector] Input  → name='{self.input_name}', shape={self.session.get_inputs()[0].shape}")
        print(f"[DeepfakeDetector] Output → name='{self.output_name}', shape={self.session.get_outputs()[0].shape}")
        print("[DeepfakeDetector] Inference session ready.\n")

    def predict(self, spectrogram: np.ndarray) -> float:
        """
        Run deepfake detection inference on a Mel-spectrogram image.

        Parameters
        ----------
        spectrogram : np.ndarray
            Mel-spectrogram with shape (224, 224, 3) and values in [0.0, 1.0].
            This is the direct output of feature_extractor.create_mel_spectrogram().

        Returns
        -------
        float
            Confidence score in range [0.0, 100.0].
            Higher value → more likely REAL voice.
            Lower value  → more likely FAKE/deepfake voice.
        """
        if spectrogram.shape != (224, 224, 3):
            raise ValueError(
                f"Invalid spectrogram shape: {spectrogram.shape}. "
                "Expected (224, 224, 3)."
            )

        # --- Step 1: HWC (224, 224, 3) → CHW (3, 224, 224) ---
        chw = np.transpose(spectrogram, (2, 0, 1))

        # --- Step 2: Add batch dimension → (1, 3, 224, 224) ---
        batch = np.expand_dims(chw, axis=0)

        # --- Step 3: Cast to float32 (required by ONNX Runtime) ---
        batch = batch.astype(np.float32)

        # --- Step 4: Run inference ---
        raw_output = self.session.run(
            [self.output_name],
            {self.input_name: batch}
        )

        # --- Step 5: Extract Probability ---
        # The model exported via export_onnx.py already includes nn.Sigmoid()
        sigmoid_score = float(raw_output[0][0][0])

        # --- Step 6: Scale to 0–100 confidence ---
        confidence = round(sigmoid_score * 100.0, 2)
        return confidence


# ---------------------------------------------------------------------------
# Quick Test Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  Call-Guardian — DeepfakeDetector Inference Test")
    print("=" * 55)

    # 1. Instantiate the detector
    detector = DeepfakeDetector()

    # 2. Simulate a spectrogram (224, 224, 3) with random data
    #    In production this comes from feature_extractor.create_mel_spectrogram()
    print("[Test] Generating dummy spectrogram input (224, 224, 3)...")
    dummy_spectrogram = np.random.rand(224, 224, 3).astype(np.float32)

    # 3. Run prediction
    score = detector.predict(dummy_spectrogram)

    # 4. Display result
    print(f"\n[Result] Deepfake Confidence Score: {score:.2f} / 100")
    if score >= 50.0:
        verdict = "✅ REAL voice (likely genuine)"
    else:
        verdict = "⚠️  FAKE voice (possible deepfake)"
    print(f"[Result] Verdict: {verdict}")
    print("=" * 55)
