FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN pip install --upgrade pip

RUN pip uninstall tensorflow keras -y

RUN pip install tf-nightly

RUN pip install sklearn

RUN pip install --upgrade cython

RUN pip install pandas