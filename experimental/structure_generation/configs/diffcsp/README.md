# DiffCSP:


### Training
```bash
# multi-gpu training, we use 4 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3" structure_generation/train.py -c structure_generation/configs/diffcsp/diffcsp_mp20.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/diffcsp/diffcsp_mp20.yaml
```

### Validation
```bash
python structure_generation/train.py -c structure_generation/configs/diffcsp/diffcsp_mp20.yaml Global.do_eval=True Global.do_train=False
```

### Testing
```bash
python structure_generation/train.py -c structure_generation/configs/diffcsp/diffcsp_mp20.yaml Global.do_test=True Global.do_train=False
```

### Sample
```bash
python structure_generation/sample.py
```
