from openai import OpenAI

import os

client = OpenAI(
    # generate you API key on openAI, then input below
    api_key="OPENAI_API_KEY"
)
from docx import Document

def transcribe_audio(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file:
        transcription = client.audio.transcriptions.create(model="whisper-1", file= audio_file)
    return transcription['text']



