# Dobble Inference image for Jetson Nano

## Build
```
sudo docker build -t fuzzylabs/dobble-jetson-nano .
```

## Run
```
sudo docker run --rm -it --runtime nvidia -v model:/dobble/model $additional_flags fuzzylabs/dobble-jetson-nano $source [$output]
```

The default entrypoint is the inference script, that requires a source (an image file, a video file or a camera device). The output is optional

Additional flags:
* `--device $DEV` e.g. `--device /dev/video0` -- to mount camera devices used as a source
* `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v /tmp/argus_socket:/tmp/argus_socket` -- pass DISPLAY environment variables and Xorg related files for GUI forwarding. 
* `-v $HOST_FILES:$CONTAINER_FILES` -- to mount files/directories, such as a directory with source files, and a directory to persist outputs to

An example of a full inference command:
```
docker run --runtime nvidia -it --rm --device /dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v /tmp/argus_socket:/tmp/argus_socket -v `pwd`/examples:./examples fuzzylabs/dobble-jetson-nano ./examples/source.mp4 ./example/
```

To start a shell in the container
```
docker run --runtime nvidia -it --rm $additional_flags --entrypoint /bin/bash fuzzylabs/dobble-jetson-nano
```