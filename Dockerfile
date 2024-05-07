FROM python:3.9

WORKDIR /code

COPY . .

RUN pip install -r requirements.txt

ENV PYTHONPATH=/code

EXPOSE 5000

CMD ["python3", "src/python/App.py", "--host=0.0.0.0"]
