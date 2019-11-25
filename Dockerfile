FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

# copy requirements.txt
COPY requirements.txt .

RUN while read requirement; do pip install $requirement; done < requirements.txt
