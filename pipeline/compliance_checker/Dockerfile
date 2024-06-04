FROM python:3.9-bullseye

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

#Copy the scripts
COPY scripts /root/bin/scripts
RUN chmod -R +x /root/bin/scripts/compliance_check.sh
RUN chmod -R +x /root/bin/scripts/installhelper.sh
ENV PATH="/root/bin/scripts:${PATH}"
