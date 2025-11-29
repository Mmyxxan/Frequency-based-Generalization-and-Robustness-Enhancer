# my_project/config.py
from yacs.config import CfgNode as CN


_C = CN()

# MODEL
_C.MODEL = CN()

# DATASET
_C.DATASET = CN()

# DATALOADER
_C.DATALOADER = CN()

# TRAINER
_C.TRAINER = CN()

# EVALUATOR
_C.EVALUATOR = CN()
