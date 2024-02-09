import whisper

def Transcribe(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    print(result["text"])

# Transcribe("AMD.mp3")
# Transcribe("BRVO.mp3")
# Transcribe("Complex RD.mp3")
# Transcribe("CRAO OD.mp3")
# Transcribe("DM.mp3")
# Transcribe("Myopia, post RD repair.mp3")
# Transcribe("PEHCR, PM.mp3")
Transcribe("No view status post repair.mp3")

