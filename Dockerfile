#FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

COPY requirements.txt /requirements.txt 
RUN pip install -r /requirements.txt

RUN apt-get update && apt-get install g++ -y

COPY pointnet2 /code/pointnet2
COPY ops_pytorch /code/ops_pytorch
COPY *.py /code/

RUN cd /code/pointnet2/ && python setup.py install

RUN cd /code/ops_pytorch/fused_conv_random_k && python setup.py install
RUN cd /code/ops_pytorch/fused_conv_select_k && python setup.py install