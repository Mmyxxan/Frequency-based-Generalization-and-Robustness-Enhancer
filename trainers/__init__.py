from .trainer import *

from utils import logger

def build_trainer(cfg):
    if cfg.TRAINER.TYPE == 0:
        return StandardTrainer(cfg=cfg)
    elif cfg.TRAINER.TYPE == 1:
        return BaselineTester(cfg=cfg)
    elif cfg.TRAINER.TYPE == 2:
        return JaFRTrainer(cfg=cfg)
    elif cfg.TRAINER.TYPE == 3:
        return RoHLTrainer(cfg=cfg)
    elif cfg.TRAINER.TYPE == 4:
        return NTIRETrainer(cfg=cfg)
    else:
        logger.error(f"Unknown trainer type: {cfg.TRAINER.TYPE}")
        raise ValueError(f"Unknown trainer type: {cfg.TRAINER.TYPE}")

# workflow: load dataset & model, train, test
