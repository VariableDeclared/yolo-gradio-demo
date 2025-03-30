FROM arm64v8/ubuntu:22.04
RUN apt update && apt install -y python3 python3-pip
COPY app.py requirements.txt /yolo-demo/
RUN pip install -r /yolo-demo/requirements.txt
