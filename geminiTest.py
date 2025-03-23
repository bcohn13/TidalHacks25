import speech_recognition as sr
from pydub import AudioSegment
import os
import yt_dlp
from google import genai
from google.genai import types
import httpx


def download_audio_from_youtube(video_url, output_file):
    # Set up options for yt-dlp to download the best audio
    ydl_opts = {
        'format': 'bestaudio/best',  # Get the best available audio
        'outtmpl': 'downloaded_audio.%(ext)s',  # Output file name (without the extension)
        'quiet': True  # Suppress download output
    }
    
    # Download the audio using yt-dlp
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print("Downloading audio...")
        ydl.download([video_url])

    print("Download completed.")
    audio = AudioSegment.from_file('downloaded_audio.webm', format="webm")
        
        # Export the audio to a .wav file
    audio.export(output_file, format="wav")

video_url = 'https://youtu.be/LOV_BwQfOqI?si=zGpO0jpMEXrryvWJ'  # Replace with your video URL
output_file = "audio_output.wav"  # Desired output WAV file name

download_audio_from_youtube(video_url, output_file)

    # Convert the downloaded audio to WAV if it's not already in that format
def transcribe_audio_from_chunks(file_path):
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    audio = AudioSegment.from_wav(file_path)
    
    # Split the audio into 1-minute chunks (60000ms)
    chunk_length_ms = 60000  # 1 minute = 60 seconds = 60000 milliseconds
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    # Process each chunk
    for i, chunk in enumerate(chunks):
        # Save the chunk as a temporary file
        chunk_file = f"chunk_{i}.wav"
        chunk.export(chunk_file, format="wav")
        print(f"Processing chunk {i + 1}/{len(chunks)}...")

        try:
            # Load the chunk into the recognizer
            with sr.AudioFile(chunk_file) as source:
                print(f"Recognizing audio in chunk {i + 1}...")
                audio_data = recognizer.record(source)
            
            # Transcribe the chunk using Google's Speech API
            text = recognizer.recognize_google(audio_data)
            print(f"Transcription for chunk {i + 1}:")
            print(text)  # Print the transcribed text for this chunk
            print("-" * 50)  # Separator between chunks

        except sr.RequestError as e:
            print(f"Error with the recognition service in chunk {i + 1}: {e}")
        except sr.UnknownValueError:
            print(f"Google Speech Recognition could not understand the audio in chunk {i + 1}.")
        except Exception as e:
            print(f"An unexpected error occurred in chunk {i + 1}: {e}")
        
        # Clean up by deleting the temporary chunk file after processing
        if os.path.exists(chunk_file):
            os.remove(chunk_file)  # Remove the temporary chunk file
            print(f"Deleted temporary file: {chunk_file}")

# Path to your WAV file
audio_path = "audio_output.wav"
transcribe_audio_from_chunks(audio_path)
