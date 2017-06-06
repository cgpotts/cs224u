# CS224u: Natural Language Understanding

Code for [the Stanford course](http://web.stanford.edu/class/cs224u/)

# Instructors

* [Bill MacCartney](http://nlp.stanford.edu/~wcmac/)
* [Christopher Potts](http://web.stanford.edu/~cgpotts/)

# Instructions for use with Docker

Docker image comes with all course requirements pre-installed and starts a jupyter server that is ready to use at http://0.0.0.0:8888. If you want to use off the shelf version, skip to **Step 2**.

* Step 1: Build image (One time thing)
Customize the Dockerfile to suit your needs and
```
docker build . -t <your image name>:latest
```

* Step 2: 

```
docker run -it --rm -p 8888:8888 -v $PWD:/notebooks -v ~<your usename here>/Downloads:/data --name cs224_container ai2160/cs224
```

Download and unzip new datasets to default **~/Downloads** folder. This folder is available in the jupyter notebook as **/data**

If you are using a custom image built in **Step 1**, replace "ai2160/cs224" with "your_image_name>"

