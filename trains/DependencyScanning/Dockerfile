FROM python:3.9


WORKDIR /usr/src/app

# LABEL
LABEL "envs"="[{\"name\": \"DB_NAME\",\"type\": \"text\",\"required\": true},{\"name\": \"DATA_SOURCE_USERNAME\",\"type\": \"text\",\"required\": true},{\"name\": \"DATA_SOURCE_HOST\",\"type\": \"text\",\"required\": true},{\"name\": \"DATA_SOURCE_PORT\",\"type\": \"number\",\"required\": true},{\"name\": \"DATA_SOURCE_PASSWORD\",\"type\": \"password\",\"required\": true},{\"name\": \"STATION_NAME\",\"type\": \"text\",\"required\": true}]"

COPY . .

RUN pip install -r requirements.txt

CMD [ "python", "main.py" ]
