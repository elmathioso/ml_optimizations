# from https://stackoverflow.com/questions/59693655/building-tensorflow-from-source-with-docker

# docker run -it --rm -w /tensorflow_src -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" tensorflow/tensorflow:devel bash

FROM tensorflow/tensorflow:devel

RUN python -m pip uninstall tensorflow
WORKDIR /tensorflow_src
RUN apt update && apt install -y clang
RUN git pull
RUN git fetch --tags
RUN git checkout v2.9.0
RUN ./configure
RUN bazel build //tensorflow/tools/pip_package:build_pip_package --repo_env=WHEEL_NAME=tensorflow_cpu_2.9.0 --config=opt
RUN ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tmp/tf_pkg
RUN python -m pip install ./tmp/tf_pkg/*.whl
WORKDIR /
RUN python -c "import tensorflow as tf"

