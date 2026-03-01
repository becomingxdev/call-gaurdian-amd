from audio_capture import capture_system_audio
audio = capture_system_audio(2)
print(type(audio), len(audio) if audio is not None else None)