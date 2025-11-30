from .trainer import StandardTrainer

import logging
logger = logging.getLogger(__name__)

def build_trainer(cfg):
    if cfg.TRAINER.TYPE == 0:
        return StandardTrainer(cfg=cfg)
    else:
        logger.error(f"Unknown trainer type: {cfg.TRAINER.TYPE}")
        raise ValueError(f"Unknown trainer type: {cfg.TRAINER.TYPE}")

# workflow: load dataset & model, train, test
