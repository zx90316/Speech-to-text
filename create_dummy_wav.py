import wave
import struct

# Mono, 16kHz, 1 second silence
sample_rate = 16000
duration = 1.0
file_name = "umdje-u6te1.wav"

with wave.open(file_name, "w") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    n_frames = int(sample_rate * duration)
    data = struct.pack("<" + "h" * n_frames, *([0] * n_frames))
    wav_file.writeframes(data)

print(f"Created {file_name}")
