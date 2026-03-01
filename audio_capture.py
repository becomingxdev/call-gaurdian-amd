import sounddevice as sd
import numpy as np
import librosa

# Device 17: Stereo Mix (Realtek HD Audio Stereo input) — Windows WDM-KS
# Records at 44100 Hz (native device rate), then resamples to 16000 Hz for the model.
LOOPBACK_DEVICE_INDEX = 17
recording_rate = 44100
TARGET_RATE = 16000


def find_stereo_mix_device():
    """Dynamically find the record device index for 'Stereo Mix'."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "Stereo Mix" in dev['name']:
            return i
    # Return 17 as a legacy fallback only if found in name-check, 
    # but the requirement says raise RuntimeError if not found.
    raise RuntimeError("Critical: 'Stereo Mix' device not found. Ensure it is enabled in Windows Sound Settings.")


def capture_system_audio(duration_sec=2):
    """
    Captures system audio using Stereo Mix at 44100 Hz,
    sanitizes, then resamples to 16000 Hz using librosa.
    """
    print("Listening to system audio...")
    
    try:
        device_id = find_stereo_mix_device()
        
        audio = sd.rec(
            int(duration_sec * recording_rate),
            samplerate=recording_rate,
            channels=1,
            dtype='float32',
            device=device_id,
        )
        sd.wait()

        # 1. Sanitize BEFORE resampling
        # Replace NaNs/Infs and clip to valid PCM range [-1, 1]
        audio_raw = np.nan_to_num(audio.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
        audio_raw = np.clip(audio_raw, -1.0, 1.0)

        # 2. Resample
        audio_resampled = librosa.resample(
            audio_raw,
            orig_sr=recording_rate,
            target_sr=TARGET_RATE,
        )

        # 3. Sanitize AFTER resampling
        # librosa resampling can occasionally introduce artifacts/overshoot
        audio_final = np.nan_to_num(audio_resampled, nan=0.0, posinf=0.0, neginf=0.0)
        audio_final = np.clip(audio_final, -1.0, 1.0)

        print(f"Audio captured: {len(audio_final)} samples @ {TARGET_RATE} Hz")
        return audio_final

    except Exception as e:
        print(f"[ERROR] Audio capture failed: {e}")
        return None


if __name__ == "__main__":
    audio_buffer = capture_system_audio()

    if audio_buffer is not None:
        print(f"Total samples : {len(audio_buffer)}")
        print(f"Max amplitude : {np.max(np.abs(audio_buffer)):.4f}")
    else:
        print("Audio capture failed.")