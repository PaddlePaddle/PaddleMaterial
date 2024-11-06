


PYTHONPATH=$PWD python -m paddle.distributed.launch --gpus="0,1" property_prediction/train.py
