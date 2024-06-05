FROM python:3.12-alpine

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

#Copy the scripts
COPY scripts /root/bin/scripts
RUN chmod -R +x /root/bin/scripts/decision.sh
RUN chmod -R +x /root/bin/scripts/decision_badge.sh
ENV PATH="/root/bin/scripts:${PATH}"
