#!/bin/bash
cd /home/aistudio/plum/ppmat/PaddleMaterials

# Enable CUDA MPS for GPU sharing
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
nvidia-cuda-mps-control -d 2>/dev/null || true
sleep 1

nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_qm9_Cv.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_Cv.log 2>&1 &
echo "Started spherenet_qm9_Cv, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_Cv.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_qm9_G.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_G.log 2>&1 &
echo "Started spherenet_qm9_G, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_G.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_qm9_H.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_H.log 2>&1 &
echo "Started spherenet_qm9_H, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_H.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_qm9_U.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_U.log 2>&1 &
echo "Started spherenet_qm9_U, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_U.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_qm9_U0.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_U0.log 2>&1 &
echo "Started spherenet_qm9_U0, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_U0.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_qm9_alpha.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_alpha.log 2>&1 &
echo "Started spherenet_qm9_alpha, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_alpha.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_qm9_gap.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_gap.log 2>&1 &
echo "Started spherenet_qm9_gap, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_gap.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_qm9_homo.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_homo.log 2>&1 &
echo "Started spherenet_qm9_homo, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_homo.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_qm9_lumo.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_lumo.log 2>&1 &
echo "Started spherenet_qm9_lumo, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_lumo.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_qm9_mu.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_mu.log 2>&1 &
echo "Started spherenet_qm9_mu, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_mu.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_qm9_r2.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_r2.log 2>&1 &
echo "Started spherenet_qm9_r2, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_r2.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_qm9_zpve.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_zpve.log 2>&1 &
echo "Started spherenet_qm9_zpve, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_qm9_zpve.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_md17_aspirin.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_aspirin.log 2>&1 &
echo "Started spherenet_md17_aspirin, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_aspirin.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_md17_benzene_old.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_benzene_old.log 2>&1 &
echo "Started spherenet_md17_benzene_old, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_benzene_old.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_md17_ethanol.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_ethanol.log 2>&1 &
echo "Started spherenet_md17_ethanol, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_ethanol.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_md17_malonaldehyde.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_malonaldehyde.log 2>&1 &
echo "Started spherenet_md17_malonaldehyde, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_malonaldehyde.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_md17_naphthalene.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_naphthalene.log 2>&1 &
echo "Started spherenet_md17_naphthalene, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_naphthalene.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_md17_salicylic.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_salicylic.log 2>&1 &
echo "Started spherenet_md17_salicylic, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_salicylic.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_md17_toluene.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_toluene.log 2>&1 &
echo "Started spherenet_md17_toluene, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_toluene.log"
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH="/home/aistudio/plum/ppmat/PaddleMaterials:$PYTHONPATH" python3 -u property_prediction/train.py -c property_prediction/configs/spherenet/spherenet_md17_uracil.yaml' > /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_uracil.log 2>&1 &
echo "Started spherenet_md17_uracil, log: /home/aistudio/plum/ppmat/PaddleMaterials/output/spherenet_md17_uracil.log"

echo "=== All 20 tasks launched ==="
echo "Check logs: ls -la output/*.log"
echo "Monitor: tail -f output/spherenet_qm9_homo.log"