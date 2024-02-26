#!/bin/bash

docker exec -it zoneminder rm -r /usr/local/lib/python3.8/dist-packages/cv2
docker exec -it zoneminder pip install --upgrade pip

docker compose restart

docker exec -it zoneminder pip cache purge
docker exec -it zoneminder pip install opencv-python onnxruntime-gpu==1.15.1