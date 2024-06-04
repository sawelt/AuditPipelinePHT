FROM alpine:latest

#Copy the scripts
COPY scripts /root/bin/scripts
RUN chmod -R +x /root/bin/scripts/disallowlist_check.sh
ENV PATH="/root/bin/scripts:${PATH}"
