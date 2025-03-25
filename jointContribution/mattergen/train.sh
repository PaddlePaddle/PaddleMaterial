

# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

#--------------------------- mp20---------------------------------------#  
# PYTHONPATH=$PWD HYDRA_FULL_ERROR=1 python scripts/run.py 
# PYTHONPATH=$PWD HYDRA_FULL_ERROR=1 python -m paddle.distributed.launch --gpus="1,2,3,4" scripts/run.py 


#--------------------------- alex_mp20----------------------------------#
# PYTHONPATH=$PWD HYDRA_FULL_ERROR=1 python scripts/run.py data_module=alex_mp_20
# PYTHONPATH=$PWD HYDRA_FULL_ERROR=1 python -m paddle.distributed.launch --gpus="1,2,3,4" scripts/run.py  data_module=alex_mp_20


# --------------------------- 2d_30k----------------------------------#
# PYTHONPATH=$PWD HYDRA_FULL_ERROR=1 python scripts/run.py data_module=2d_30k
# PYTHONPATH=$PWD HYDRA_FULL_ERROR=1 python -m paddle.distributed.launch --gpus="4,5,6,7" scripts/run.py data_module=2d_30k


# --------------------------- 2d_30k add mean distance----------------------------------#
# PYTHONPATH=$PWD HYDRA_FULL_ERROR=1 python scripts/run.py --config-name=default_2d_md
# PYTHONPATH=$PWD HYDRA_FULL_ERROR=1 python -m paddle.distributed.launch --gpus="4,5,6,7" scripts/run.py --config-name=default_2d_md
