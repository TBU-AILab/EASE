FROM python:3.11

WORKDIR /app

# Install pip packages 
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Port is exposed in the docker-compose.yml file
#CMD ["uvicorn", "fastapi1_main:app", "--host", "0.0.0.0", "--port", "8086", "--reload", "--reload-exclude", "out_task"]
CMD ["uvicorn", "fastapi1_main:app", "--host", "0.0.0.0", "--port", "8086"]