import librosa
import pandas as pd

'''
path is audio file path
pipe is initiated model pipeline
Both are required 

returns a processed dataframe with time stamps
'''
def transcribe(path, pipe):
    # Load audio file using librosa
    audio_file_path = path
    audio_data, sample_rate = librosa.load(audio_file_path, sr=None)  

    # Resample audio data to 16,000 Hz
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate

    result = pipe(audio_data, return_timestamps=True)
    df = pd.DataFrame(result["chunks"])
    df[['start_time', 'end_time']] = pd.DataFrame(df['timestamp'].tolist(), index=df.index)

    return df