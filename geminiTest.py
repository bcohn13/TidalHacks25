import speech_recognition as sr
from pydub import AudioSegment
import os

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
audio_path = "path_to_your_audio.wav"
transcribe_audio_from_chunks(audio_path)