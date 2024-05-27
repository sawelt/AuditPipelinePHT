FROM python:3.6

WORKDIR /

# LABEL
LABEL "envs"="[{\"name\":\"FHIR_SERVER\",\"type\":\"string\",\"required\":true},{\"name\":\"FHIR_PORT\",\"type\":\"number\",\"required\":true}]"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./main.py" ]