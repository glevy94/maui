
import yt_dlp

'''

Takes in a list of youtube urls 
Saves audio from youtube videos in WAV format at 16 KHz sample rate 

'''
def download_audio(urls):

    # yt_dlp options
    ydl_opts = {
        'format': 'bestaudio/best',  
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Convert to WAV format, lossless format
        }],
        'postprocessor_args': [
            '-ar', '16000'  # Set the audio sample rate to 16,000 Hz for use with whisper
        ],
        'outtmpl': 'audio_files/%(title)s.%(ext)s', 
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)
