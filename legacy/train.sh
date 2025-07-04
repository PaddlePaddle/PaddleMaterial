
export PYTHONPATH=$PWD
python -m paddle.distributed.launch --gpus="0,1,2,3" property_prediction/train.py -c property_prediction/configs/dimenet_mp18.yaml
