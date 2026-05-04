FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg libsndfile1

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install voxcpm fastapi uvicorn soundfile

# download model early (important!)
RUN python3 -c "from voxcpm import VoxCPM; VoxCPM.from_pretrained('openbmb/VoxCPM2')"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
