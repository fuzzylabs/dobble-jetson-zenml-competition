FROM dustynv/jetson-inference:r32.7.1

ENV LD_PRELOAD /usr/lib/aarch64-linux-gnu/libgomp.so.1

WORKDIR /dobble-jetson-nano

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT [ "python3", "detect-dobble.py" ]