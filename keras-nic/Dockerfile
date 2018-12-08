# Testing creation of container with keras, numpy, pandas, etc..

FROM tensorflow/tensorflow:latest-gpu
MAINTAINER "Carlos Antonetti"

RUN apt-get update

RUN pip install --upgrade pip

RUN apt-get -y install libgtk2.0-dev

# Install other useful Python packages using pip
RUN pip install --upgrade ipython && \
	pip install \
		scikit-learn \
		pandas \
		numpy \
		matplotlib \
		nltk \
		&& \
	python -m ipykernel.kernelspec


# Jupyter and Tensorboard ports
EXPOSE 6006

WORKDIR "/root"
CMD ["/bin/bash"]
