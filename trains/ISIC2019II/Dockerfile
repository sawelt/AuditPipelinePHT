FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN apt-get update && apt-get install -y git
WORKDIR /usr/src/app

LABEL "envs"="[{\"name\":\"FHIR_SERVER\",\"type\":\"string\",\"required\":true},{\"name\":\"FHIR_PORT\",\"type\":\"number\",\"required\":true},{\"name\":\"BATCH_SIZE\",\"type\":\"number\",\"required\":true},{\"name\":\"LR\",\"type\":\"string\",\"required\":true},{\"name\":\"WEIGTH_DECAY\",\"type\":\"number\",\"required\":true}]"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./main.py" ]