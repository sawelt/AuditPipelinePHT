FROM python:3.11.7-slim


WORKDIR /usr/src/app

# LABEL
LABEL "envs"="[{\"name\": \"DB_NAME\",\"type\": \"text\",\"required\": true},{\"name\": \"DATA_SOURCE_USERNAME\",\"type\": \"text\",\"required\": true},{\"name\": \"DATA_SOURCE_HOST\",\"type\": \"text\",\"required\": true},{\"name\": \"DATA_SOURCE_PORT\",\"type\": \"number\",\"required\": true},{\"name\": \"DATA_SOURCE_PASSWORD\",\"type\": \"password\",\"required\": true},{\"name\": \"STATION_NAME\",\"type\": \"text\",\"required\": true}]"

COPY . .

CMD [ "python", "main.py" ]
