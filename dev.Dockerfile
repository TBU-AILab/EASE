FROM python:3.11

WORKDIR /app

# Install pip packages 
COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get -y install cmake
RUN apt install -y libgl1-mesa-glx & apt -y install ffmpeg
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Port is exposed in the docker-compose.yml file
CMD ["uvicorn", "fastapi1_main:app", "--host", "0.0.0.0", "--port", "8086"]