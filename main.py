import os
import pandas as pd
from video_to_audio import download_audio
from model_pipe import init_model
from transcribe import transcribe

urls = ['https://www.youtube.com/watch?v=4tOiX5j3_ek']
new_audio = True

if new_audio:
    download_audio(urls)

# Initialize pipeline of whisper model
pipe = init_model()

# Path to folder with audio files
folder_path = 'audio_files/'

for filename in os.listdir('audio_files/'):
    audio_path = os.path.join(folder_path, filename)
    df = transcribe(path = audio_path, pipe = pipe)

    base_name = os.path.splitext(filename)[0]
    csv_filename = f"{base_name}.csv"
    df.to_csv(csv_filename, index=False)
    
    print(f"Audio from {filename} saved to CSV {csv_filename}")

