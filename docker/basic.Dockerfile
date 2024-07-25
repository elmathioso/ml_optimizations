FROM tensorflow/tensorflow:2.9.0

# Install libs and entrypoint

RUN python -m pip install tensorflow_datasets

ADD ./src /src

ENTRYPOINT python /src/train_model.py