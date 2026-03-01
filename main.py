import time
import sys
import numpy as np
from audio_capture import capture_system_audio
from feature_extractor import create_mel_spectrogram
from inference_engine import DeepfakeDetector

def run_pipeline():
    """
    Main real-time pipeline for Call-Guardian.
    Captures system audio, processes it into a spectrogram, 
    and runs deepfake detection inference.
    """
    print("=" * 60)
    print("        CALL-GUARDIAN: REAL-TIME DEEPFAKE DETECTOR")
    print("=" * 60)
    
    # 1. Initialise the AI Model (DirectML/AMD accelerated)
    try:
        detector = DeepfakeDetector()
    except Exception as e:
        print(f"Critial Error initialising detector: {e}")
        return

    print("\n[Status] Pipeline active. Press Ctrl+C to stop.")
    print("-" * 60)

    try:
        while True:
            # 2. Capture 2 seconds of system audio (Loopback)
            # This captures what is currently playing through the speakers.
            audio_buffer = capture_system_audio(duration_sec=2)
            
            if audio_buffer is not None and len(audio_buffer) > 0:
                # 3. Convert raw audio to scaled Mel-spectrogram (224, 224, 3)
                visual_data = create_mel_spectrogram(audio_buffer, sample_rate=16000)
                
                # 4. Run AI Inference
                confidence = detector.predict(visual_data)
                
                # 5. Determine Verdict
                # Confidence > 50 means likely human/real.
                if confidence >= 40.0:
                    verdict = "✅ HUMAN / REAL"
                    status_color = "REAL"
                else:
                    verdict = "⚠️ AI / DEEPFAKE LIKELY"
                    status_color = "FAKE"
                
                # 6. Output Results
                print(f"\n>>> Deepfake Confidence: {confidence:.2f} %")
                print(f">>> Verdict: {verdict}")
                print("-" * 60)
            else:
                print("[Warning] No audio detected. Ensure system audio is playing.")
            
            # 7. Cool-down before next capture
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n[Status] Stopping pipeline...")
        print("[Status] Call-Guardian deactivated. Goodbye.")
        sys.exit(0)

if __name__ == "__main__":
    run_pipeline()
