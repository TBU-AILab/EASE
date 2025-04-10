FROM python:3.11

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get -y install cmake
RUN apt apt install -y libgl1-mesa-glx
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

RUN mkdir /app/task_out
COPY ./fastapi1_main.py /app/fastapi1_main.py
COPY ./fopimt /app/fopimt

EXPOSE 8086
CMD ["uvicorn", "fastapi1_main:app", "--host", "0.0.0.0", "--port", "8086"]
