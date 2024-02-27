FROM dlandon/zoneminder.machine.learning:latest

WORKDIR /

# COPY . .

RUN rm -r /usr/local/lib/python3.8/dist-packages/cv2
RUN pip install --upgrade pip \
    pip cache purge \
    pip install opencv-python onnxruntime-gpu==1.15.1

EXPOSE 80 443 9000