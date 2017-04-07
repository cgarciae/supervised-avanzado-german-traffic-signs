FROM python:2.7

ADD requirements.txt requirements.txt

RUN pip install -r requirements.txt

CMD jupyter notebook \
  --allow-root \
  --port=8888 \
  --ip=0.0.0.0 \
  --NotebookApp.token= \
  .
