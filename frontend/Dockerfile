FROM python:3.9

WORKDIR /frontend

COPY requirements.txt requirements.txt 

RUN pip3 install -r requirements.txt

COPY . .

# CMD ["streamlit", "run", "entrypoint.py", "--server.port", "1235"]