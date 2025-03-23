import speech_recognition as sr
from pydub import AudioSegment
import os
import yt_dlp
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
from pytube import Search

genai.configure(api_key=os.environ.get('api_key'))

def youtube_search_urls(search_string):
    """
    Performs a YouTube search for each string in a list and returns a list of URLs.
    """
    results = []
    try:
        s = Search(search_string)
        results = [video.watch_url for video in s.results]
    except Exception as e:
        print(f"Error searching for '{search_string}': {e}")
        results = []
    return results

def download_audio_from_youtube(video_url, output_file):
    ydl_opts = {'format': 'bestaudio/best', 'outtmpl': 'downloaded_audio.%(ext)s', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print("Downloading audio...")
        ydl.download([video_url])
    print("Download completed.")
    audio = AudioSegment.from_file('downloaded_audio.webm', format="webm")
    audio.export(output_file, format="wav")
    if os.path.exists('downloaded_audio.webm'):
        os.remove('downloaded_audio.webm')
        print("Deleted temporary file: downloaded_audio.webm")

def transcribe_audio_from_chunks(file_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(file_path)
    chunk_length_ms = 60000
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    textstr = ""
    for i, chunk in enumerate(chunks):
        chunk_file = f"chunk_{i}.wav"
        chunk.export(chunk_file, format="wav")
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        try:
            with sr.AudioFile(chunk_file) as source:
                print(f"Recognizing audio in chunk {i + 1}...")
                audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            print(f"Transcription for chunk {i + 1}:")
            print(text)
            textstr += text
            print("-" * 50)
        except sr.RequestError as e:
            print(f"Error with the recognition service in chunk {i + 1}: {e}")
        except sr.UnknownValueError:
            print(f"Google Speech Recognition could not understand the audio in chunk {i + 1}.")
        except Exception as e:
            print(f"An unexpected error occurred in chunk {i + 1}: {e}")
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            print(f"Deleted temporary file: {chunk_file}")
    print(textstr)
    return textstr

request = youtube_search_urls("Dylan A. Shell Texas A&M")
for vid in request:
    video_url = vid
    output_file = "audio_output.wav"
    download_audio_from_youtube(video_url, output_file)
    audio_path = "audio_output.wav"
    data = transcribe_audio_from_chunks(audio_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"Deleted temporary file: {audio_path}")

    prompt = """
    From the audio transcript, extract the key topics discussed. 
    Organize the information in a way that is easily understandable.
    If the data can be represented as a graph, format it as:
    graph_type:<graph type>
    x_column:category
    y_column:value
    data:
    category,value
    A,10
    B,25
    C,15
    D,30

    If the data is better represented as a table, format it as a table.
    If the data is best represented as a list, format it as a list.
    If the data is best represented as a network graph, format it as a network graph.
    If the data is not able to be represented as a graph, table, list, or network graph, then summarize the transcript in paragraph format.
    Display nothing else but the formatted information.
    """

    response = genai.GenerativeModel("gemini-2.0-flash").generate_content([data, prompt])
    print(response.text)

    if "graph_type:" in response.text and "data:" in response.text:
        def parse_and_plot(input_string):
            parts = input_string.split("data:")
            metadata = parts[0].strip().split("\n")
            data_csv = parts[1].strip()
            metadata_dict = {}
            for item in metadata:
                key, value = item.split(":")
                metadata_dict[key.strip()] = value.strip()
            graph_type = metadata_dict.get("graph_type", "line")
            x_col = metadata_dict.get("x_column")
            y_col = metadata_dict.get("y_column")
            try:
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
            except:
                print("Gemini response was not in a parsable format for a seaborn graph.")

        parse_and_plot(response.text)
    else:
        print("Gemini response was not in the correct format for a seaborn graph. Outputting the response:")
        print(response.text)