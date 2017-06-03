# CS224u: Natural Language Understanding

Code for [the Stanford course](http://web.stanford.edu/class/cs224u/)

# Instructors

* [Bill MacCartney](http://nlp.stanford.edu/~wcmac/)
* [Christopher Potts](http://web.stanford.edu/~cgpotts/)

# Instructions for use with Docker

Docker image comes with all course requirements installed and starts a jupyter server that is ready to use.

* Step 1: Build image (One time thing)
```
docker build . -t cs224:latest
```
This will soon be unnecessary as the image will be hosted on docker hub. 

* Step 2: 
```
docker run -it --rm -p 8888:8888 -v $PWD:/notebooks --name cs224_container cs224
```

