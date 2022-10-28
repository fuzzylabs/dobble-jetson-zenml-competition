FROM dustynv/jetson-inference:r35.1.0

# TODO check that this hacks are not needed anymore!!
# RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
#     dpkg-reconfigure --frontend=noninteractive locales && \
#     update-locale LANG=en_US.UTF-8

# ENV LANG en_US.UTF-8 
# ENV OPENBLAS_CORETYPE ARMV8

WORKDIR /dobble-jetson-nano

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT [ "python3", "detect-dobble.py" ]