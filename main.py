# should implement summary writer tensorboard
# should add drop out config and in classifier
# write config for each experiment and then try running
# write code to load different datasets

import argparse
import torch

from utils import set_up_logger, set_random_seed, collect_env_info
from configs import get_cfg_defaults
from trainers import build_trainer

import logging
logger = logging.getLogger(__name__)

def print_args(args, cfg):
    logger.info("***************")
    logger.info("** Arguments **")
    logger.info("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        logger.info("{}: {}".format(key, args.__dict__[key]))
    logger.info("************")
    logger.info("** Config **")
    logger.info("************")
    logger.info(cfg)

def reset_cfg(cfg, args):
    if args.data_dir:
        cfg.DATASET.DATA_DIR = args.data_dir

    if args.output_dir:
        cfg.MODEL.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.MODEL.RESUME = args.resume

    if args.seed:
        cfg.TRAINER.SEED = args.seed

    if args.transforms:
        cfg.TRANSFORM.TRANSFORMS = args.transforms

    if args.trainer_type:
        cfg.TRAINER.TYPE = args.trainer_type

    if args.model_name:
        cfg.MODEL.MODEL_NAME = args.model_name

    if args.no_tf_test:
        cfg.TRANSFORM.NO_TRANSFORM_TEST = args.no_tf_test

    if args.eval_only:
        cfg.TRAINER.IS_TRAIN = False

def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    pass

def setup_cfg(args):
    cfg = get_cfg_defaults()
    extend_cfg(cfg)

    # From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # From input arguments
    reset_cfg(cfg, args)

    # From optional input arguments
    cfg.merge_from_list(args.opts)

    if not (torch.cuda.is_available() and cfg.TRAINER.USE_CUDA):
        cfg.TRAINER.USE_CUDA = False

    cfg.freeze()

    return cfg

def main(args):
    cfg = setup_cfg(args)
    if cfg.TRAINER.SEED >= 0:
        logger.info("Setting fixed seed: {}".format(cfg.TRAINER.SEED))
        set_random_seed(cfg.TRAINER.SEED)
    set_up_logger(cfg=cfg)

    if torch.cuda.is_available() and cfg.TRAINER.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    
    print_args(args, cfg)
    logger.info("Collecting env info ...")
    logger.info("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.before_train()
        trainer.test()
        return

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="", help="path to dataset")
    parser.add_argument(
        "--output-dir", type=str, default="", help="output directory"
    )
    parser.add_argument(
        "--resume",
        type=bool,
        default=True,
        help="whether to resume from checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--trainer-type", type=int, default=0, help="type of trainer"
    )
    parser.add_argument(
        "--model-name", type=str, default="", help="name of model that should be loaded"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="evaluation only"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory",
    )
    parser.add_argument(
        "--no-tf-test",
        type=bool,
        default=False,
        help="apply no augmentation in test transform",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)

# Priority: default < config file < input arguments