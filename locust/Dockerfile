FROM locustio/locust:2.43.3

USER root

COPY AffectNet/happy /fixtures/happy
COPY locust/locustfile.py /locust/locustfile.py

RUN chown -R locust:locust /fixtures /locust/locustfile.py

USER locust
WORKDIR /locust

ENV IMAGE_PATH=/fixtures/happy

EXPOSE 8089

CMD ["-f", "/locust/locustfile.py", "--host", "http://127.0.0.1:8000", "--web-host", "0.0.0.0"]
