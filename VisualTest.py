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

video_url = 'https://youtu.be/k6U-i4gXkLM?si=0Q9UVMtRWm7fajjb'  # Replace with your video URL
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

prompt = "From the audio transcript, Display the format the you think would be appropriate"

# Use both documents in the API call

response = client.models.generate_content(
  model="gemini-2.0-flash",
  contents=[
      data,  # Unpack the list of Parts directly into contents
      prompt      # Add the text prompt
  ])

print(response.text)
def generate_graph_from_prompt(response, model_name="gemini-pro"):
    """
    Generates a graph using Gemini and Matplotlib/Seaborn based on a user prompt.
    """
    try:
        

        #Check if the response is a string, which indicates an error
        if isinstance(response, str):
            return f"Gemini API Error: {response}"

        # Check if response.parts exists and is not empty.
        if hasattr(response, "parts") and response.parts:
            data_str = "".join([part.text for part in response.parts if hasattr(part, 'text') and part.text]) # handle if part.text exists.
        else:
            return "Gemini API Error: No data received from the model."

        graph_type = "line"
        x_col = None
        y_col = None
        "bar-"
        if "bar" in prompt.lower():
            graph_type = "bar"
        elif "scatter" in prompt.lower():
            graph_type = "scatter"
        elif "histogram" in prompt.lower() or "hist" in prompt.lower():
            graph_type = "hist"

        if "x=" in prompt.lower():
            x_col = prompt.lower().split("x=")[1].split()[0].strip()
        if "y=" in prompt.lower():
            y_col = prompt.lower().split("y=")[1].split()[0].strip()

        try:
            df = pd.read_csv(io.StringIO(data_str))
        except:
            df = pd.read_csv(io.StringIO(data_str), sep='\s+')

        plt.figure(figsize=(8, 6))

        if graph_type == "line":
            if x_col and y_col:
                sns.lineplot(x=x_col, y=y_col, data=df)
            else:
                sns.lineplot(data=df)
        elif graph_type == "bar":
            if x_col and y_col:
                sns.barplot(x=x_col, y=y_col, data=df)
            else:
                sns.barplot(data=df)
        elif graph_type == "scatter":
            if x_col and y_col:
                sns.scatterplot(x=x_col, y=y_col, data=df)
            else:
                sns.scatterplot(data=df)
        elif graph_type == "hist":
            if y_col:
                sns.histplot(data=df[y_col])
            else:
                sns.histplot(data=df)

        plt.title("Generated Graph")
        plt.xlabel(x_col if x_col else "")
        plt.ylabel(y_col if y_col else "")

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        return img_base64

    except Exception as e:
        return f"An error occurred: {e}"

def display_graph(base64_image):
    if base64_image and not base64_image.startswith("Gemini API Error") and not base64_image.startswith("An error occurred"):
        display(HTML(f'<img src="data:image/png;base64,{base64_image}" />'))
    else:
        print(base64_image)
image=generate_graph_from_prompt(response)

display_graph(image)
