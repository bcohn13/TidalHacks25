import speech_recognition as sr
from pydub import AudioSegment
import os
import yt_dlp
from google import genai
from google.genai import types
import httpx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
from IPython.display import HTML, display

client = genai.Client(api_key = os.environ.get('api_key'))
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

video_url = 'https://youtu.be/SHxOQd6VTp4?si=WSdG1LygWfZcztGv'  # Replace with your video URL
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
    textstr=""
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
            print(text)
            textstr+=text # Print the transcribed text for this chunk
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
            print(f"Deleted temporary file: {chunk_file}"
                )
        print(textstr)
        return textstr

# Path to your WAV file
audio_path = "audio_output.wav"
data=transcribe_audio_from_chunks(audio_path)

prompt = "From the audio transcript, Display the format the you think would be appropriate and the data itself as a string in the format: \
graph_type:<graph type that would be appropriate>\
x_column:category \
y_column:value \
data: \
category,value \
A,10 \
B,25 \
C,15 \
D,30  \
Display noting else but the string above. The graph type must be a seaborn graph type: barplot,lineplot ect.\
"

# Use both documents in the API call

response = client.models.generate_content(
  model="gemini-2.0-flash",
  contents=[
      data,  # Unpack the list of Parts directly into contents
      prompt      # Add the text prompt
  ])

print(response.text)


def parse_and_plot(input_string):
    """Parses the input string and generates a Seaborn/Matplotlib plot."""

    parts = input_string.split("data:")
    metadata = parts[0].strip().split("\n")
    data_csv = parts[1].strip()

    metadata_dict = {}
    for item in metadata:
        key, value = item.split(":")
        metadata_dict[key.strip()] = value.strip()

    graph_type = metadata_dict.get("graph_type", "line") # default to line graph if no type specified.
    x_col = metadata_dict.get("x_column")
    y_col = metadata_dict.get("y_column")

    df = pd.read_csv(io.StringIO(data_csv))

    plt.figure(figsize=(8, 6))

    if graph_type == "bar":
        sns.barplot(x=x_col, y=y_col, data=df)
    elif graph_type == "line":
        sns.lineplot(x=x_col, y=y_col, data=df)
    elif graph_type == "scatter":
        sns.scatterplot(x=x_col, y=y_col, data=df)
    elif graph_type == "hist":
        if y_col:
            sns.histplot(data=df[y_col])
        else:
            sns.histplot(data=df)
    else:
        print("Invalid graph type.")
        return

    plt.title("Generated Graph")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

parse_and_plot(response.text)