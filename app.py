from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from voxcpm import VoxCPM
import soundfile as sf
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = VoxCPM.from_pretrained("openbmb/VoxCPM2")

@app.get("/")
def home():
    return {"message": "VoxCPM2 running"}

@app.get("/tts")
def tts(text: str):
    wav = model.generate(text=text)
    file = "output.wav"
    sf.write(file, wav, model.tts_model.sample_rate)

    with open(file, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()

    return {"audio": audio_base64}