import librosa
import numpy as np
from PIL import Image

def create_mel_spectrogram(audio_data, sample_rate=16000, target_size=(224, 224)):
    """
    Converts raw audio data into a Mel-spectrogram and resizes it to 224x224.
    Optimized: Removes Matplotlib bottleneck and uses stable dB scaling.
    """
    # 1. Generate the Mel-spectrogram (n_mels=128)
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data.astype(np.float32), 
        sr=sample_rate, 
        n_mels=128,
        fmax=8000
    )
    
    # 2. Convert power to decibels (log scale)
    # Using ref=1.0 instead of np.max for numerical stability.
    # ref=1.0 provides consistent output levels regardless of Peak Amplitude,
    # preventing silent or quiet buffers from having highly boosted noise (max-scaling).
    mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)
    
    # 3. Direct NumPy Normalization (approx. Min-Max scaling for visualization)
    # We clip to common Mel DB ranges (-80 to 0) before normalizing
    mel_spec_db = np.clip(mel_spec_db, -80.0, 0.0)
    normalized = (mel_spec_db + 80.0) / 80.0
    
    # 4. Generate RGB Input for ResNet (3 channels) via PIL
    # Convert Grayscale Log-Mel to Image and Resize
    uint8_img = (normalized * 255).astype(np.uint8)
    image = Image.fromarray(uint8_img).convert('RGB')
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Return as float32 scaled 0-1
    return np.array(image).astype(np.float32) / 255.0

# --- Quick Test ---
if __name__ == "__main__":
    # Create 2 seconds of fake "white noise" audio to test the function
    fake_audio = np.random.randn(16000 * 2) 
    visual_data = create_mel_spectrogram(fake_audio)
    
    print(f"Final AI input shape: {visual_data.shape}") 
    # Expected output: (224, 224, 3)