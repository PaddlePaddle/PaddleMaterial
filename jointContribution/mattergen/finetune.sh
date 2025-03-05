

# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

#--------------------------- mp20 single property---------------------------------------
# export PROPERTY=chemical_system
# export MODEL_PATH=checkpoints/matterten_base_mp20
# export PYTHONPATH=$PWD
# python scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]
# # python -m paddle.distributed.launch --gpus="3,4,5,6" scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]

# export PROPERTY=dft_band_gap
# export MODEL_PATH=checkpoints/matterten_base_mp20
# export PYTHONPATH=$PWD
# # python scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]
# python -m paddle.distributed.launch --gpus="3,4,5,6" scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]

# export PROPERTY=dft_mag_density
# export MODEL_PATH=checkpoints/matterten_base_mp20
# export PYTHONPATH=$PWD
# python scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]
# python -m paddle.distributed.launch --gpus="3,4,5,6" scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]

# export PROPERTY=dft_bulk_modulus
# export MODEL_PATH=checkpoints/matterten_base_mp20
# export PYTHONPATH=$PWD
# python scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]
# python -m paddle.distributed.launch --gpus="3,4,5,6" scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]

# export PROPERTY=formation_energy_per_atom
# export MODEL_PATH=checkpoints/matterten_base_mp20
# export PYTHONPATH=$PWD
# python scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]
# python -m paddle.distributed.launch --gpus="3,4,5,6" scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]

#--------------------------- mp20 multi property----------------------------------------
# export PROPERTY1=chemical_system
# export PROPERTY2=formation_energy_per_atom 
# export MODEL_PATH=checkpoints/matterten_base_mp20
# export PYTHONPATH=$PWD
# python scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 data_module.properties=["$PROPERTY1","$PROPERTY2"]
# python -m paddle.distributed.launch --gpus="3,4,5,6" scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 data_module.properties=["$PROPERTY1","$PROPERTY2"]



#--------------------------- alex_mp20 single property----------------------------------
# export PROPERTY=chemical_system
# export MODEL_PATH=checkpoints/mattergen_base
# export PYTHONPATH=$PWD
# python scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=alex_mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]
# python -m paddle.distributed.launch --gpus="3,4,5,6" scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=alex_mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]

# export PROPERTY=space_group
# export MODEL_PATH=checkpoints/mattergen_base
# export PYTHONPATH=$PWD
# python scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=alex_mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]
# python -m paddle.distributed.launch --gpus="3,4,5,6" scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=alex_mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]

# export PROPERTY=dft_mag_density
# export MODEL_PATH=checkpoints/mattergen_base
# export PYTHONPATH=$PWD
# python scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=alex_mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]
# python -m paddle.distributed.launch --gpus="3,4,5,6" scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=alex_mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]

# export PROPERTY=dft_band_gap
# export MODEL_PATH=checkpoints/mattergen_base
# export PYTHONPATH=$PWD
# python scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=alex_mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]
# python -m paddle.distributed.launch --gpus="3,4,5,6" scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=alex_mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]

# export PROPERTY=ml_bulk_modulus
# export MODEL_PATH=checkpoints/mattergen_base
# export PYTHONPATH=$PWD
# python scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=alex_mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]
# python -m paddle.distributed.launch --gpus="3,4,5,6" scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=alex_mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY data_module.properties=["$PROPERTY"]


#--------------------------- alex_mp20 multi property----------------------------------
# export PROPERTY1=chemical_system
# export PROPERTY2=energy_above_hull 
# export MODEL_PATH=checkpoints/mattergen_base
# export PYTHONPATH=$PWD
# python scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=alex_mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 data_module.properties=["$PROPERTY1","$PROPERTY2"]
# python -m paddle.distributed.launch --gpus="3,4,5,6" scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=alex_mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 data_module.properties=["$PROPERTY1","$PROPERTY2"]

# export PROPERTY1=dft_mag_density
# export PROPERTY2=hhi_score
# export MODEL_PATH=checkpoints/mattergen_base
# export PYTHONPATH=$PWD
# python scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=alex_mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 data_module.properties=["$PROPERTY1","$PROPERTY2"]
# python -m paddle.distributed.launch --gpus="3,4,5,6" scripts/finetune.py adapter.model_path=$MODEL_PATH data_module=alex_mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 data_module.properties=["$PROPERTY1","$PROPERTY2"]
