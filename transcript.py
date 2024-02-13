import whisper

def Transcribe(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    print(result["text"])

Transcribe("./recordings/AMD.mp3")
Transcribe("./recordings/BRVO.mp3")
Transcribe("./recordings/Complex RD.mp3")
Transcribe("./recordings/CRAO OD.mp3")
Transcribe("./recordings/DM.mp3")
Transcribe("./recordings/Myopia, post RD repair.mp3")
Transcribe("./recordings/PEHCR, PM.mp3")
Transcribe("./recordings/No view status post repair.mp3")

