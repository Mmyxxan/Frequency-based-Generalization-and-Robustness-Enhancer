from .fused.fused_backbone import FusedBackbone
from .fused.clip_vit import CLIPViT
from .fused.cnn_resnet50 import ResNet50
from .fused.clip_vit_fare import CLIPViT_FARE

from .baselines.clipping import clipmodel
from .baselines.cnndet import resnet50

import torch
import torch.nn as nn

import os.path as osp
import pickle
from functools import partial
from collections import OrderedDict
import shutil

from utils import mkdir_if_missing, load_checkpoint, logger

# Remember to put low frequency bias extractor first in backbone list
MODEL_TO_BACKBONES = {
    "Fused_CNN_ResNet50_CLIP_ViT_512_Concat": [ResNet50, CLIPViT],
    "Fused_CNN_ResNet50_CLIP_ViT_FARE_512_Concat": [ResNet50, CLIPViT_FARE],
    "Fused_CNN_ResNet50_CNN_ResNet50_512_Concat": [ResNet50, ResNet50],
}

def build_backbone(cfg):
    if cfg.MODEL.NAME in MODEL_TO_BACKBONES:
        logger.info(f"Loading {cfg.MODEL.NAME}")
        if "512_Concat" in cfg.MODEL.NAME:
            return FusedBackbone(backbone_list=MODEL_TO_BACKBONES[cfg.MODEL.NAME], project_dim=512, fuse_technique="Concat",
                             freeze=cfg.MODEL.BACKBONE.FREEZE, pretrained=cfg.MODEL.BACKBONE.PRETRAINED)
        elif "512_" in cfg.MODEL.NAME and "" in cfg.MODEL.NAME:
            return # implement new fusion techniques
    else:
        logger.error(f"Unknown model name: {cfg.MODEL.NAME}")
        raise ValueError(f"Unknown model name: {cfg.MODEL.NAME}")
    
class Baseline:
    def __init__(self, cfg, **kwargs):
        logger.info(f"Loading {cfg.MODEL.NAME}")
        if cfg.MODEL.NAME == "CLIPping":
            self.model = clipmodel()
        elif cfg.MODEL.NAME == "CNNDet":
            if cfg.DATASET.NUM_CLASSES == 2:
                self.model = resnet50(num_classes=1)
            else:
                logger.error(f"Incompatible # of classes of dataset: {cfg.DATASET.NUM_CLASSES}")
                raise ValueError(f"Incompatible # of classes of dataset: {cfg.DATASET.NUM_CLASSES}")
        else:
            logger.error(f"Unknown model name: {cfg.MODEL.NAME}")
            raise ValueError(f"Unknown model name: {cfg.MODEL.NAME}")

    def forward(self, x):
        return self.model(x)

    def load_checkpoint(self, cfg):
        fpath = osp.join(cfg.MODEL.MODEL_DIR, cfg.MODEL.MODEL_NAME)

        if not osp.exists(fpath):
            logger.error('File is not found at "{}"'.format(fpath))
            raise FileNotFoundError('File is not found at "{}"'.format(fpath))
        
        map_location = None if cfg.TRAINER.USE_CUDA else "cpu"

        try:
            checkpoint = torch.load(fpath, map_location=map_location)

        except UnicodeDecodeError:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(
                fpath, pickle_module=pickle, map_location=map_location
            )

        except Exception:
            logger.error('Unable to load checkpoint from "{}"'.format(fpath))
            raise

        if "state_dict" not in checkpoint:
            if "model" in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint["state_dict"]

        logger.info(f"Load {fpath} to {cfg.MODEL.TYPE}/{cfg.MODEL.NAME}")
        self.model.load_state_dict(state_dict)

class MyModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.backbone = build_backbone(cfg=cfg)

        fdim = self.backbone.out_features
        self.classifier = nn.Linear(fdim, cfg.DATASET.NUM_CLASSES)

    def forward(self, x):
        f = self.backbone(x)
        y = self.classifier(f)
        return y
    
    def load_best_model(self, directory, epoch=None):
        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        model_path = osp.join(directory, model_file)

        if not osp.exists(model_path):
            logger.error(f"No model at {model_path}")
            raise FileNotFoundError(f"No model at {model_path}")

        checkpoint = load_checkpoint(model_path)

        state_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]
        val_result = checkpoint["val_result"]
        
        logger.info(f"Load {model_path} (epoch={epoch}, val_result={val_result:.1f})")

        self.load_state_dict(state_dict)

    def resume_or_load_checkpoint(self, cfg, optimizer, scheduler):
        fpath = osp.join(cfg.MODEL.MODEL_DIR, cfg.MODEL.MODEL_NAME)

        if cfg.TRAINER.IS_TRAIN:
            if not cfg.MODEL.RESUME or not osp.exists(fpath):
                logger.info("Training model from scratch...")
                return 0
        else:
            if not osp.exists(fpath):
                logger.error('File is not found at "{}"'.format(fpath))
                raise FileNotFoundError('File is not found at "{}"'.format(fpath))        

        map_location = None if cfg.TRAINER.USE_CUDA else "cpu"

        try:
            checkpoint = torch.load(fpath, map_location=map_location)

        except UnicodeDecodeError:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(
                fpath, pickle_module=pickle, map_location=map_location
            )

        except Exception:
            logger.error('Unable to load checkpoint from "{}"'.format(fpath))
            raise

        if cfg.TRAINER.IS_TRAIN:
            logger.info(f"Resume training at checkpoint {fpath}")
            
            if optimizer is not None and "optimizer" in checkpoint.keys():
                optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info("Resume optimizer")

            if scheduler is not None and "scheduler" in checkpoint.keys():
                scheduler.load_state_dict(checkpoint["scheduler"])
                logger.info("Resume scheduler")

            start_epoch = checkpoint["epoch"]
            logger.info("Previous epoch: {}".format(start_epoch))

        state_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]
        val_result = checkpoint["val_result"]

        logger.info(f"Load {fpath} to {cfg.MODEL.TYPE}/{cfg.MODEL.NAME} (epoch={epoch}, val_result={val_result:.1f})")

        self.load_state_dict(state_dict)

        if cfg.TRAINER.IS_TRAIN:
            return start_epoch
        
        return 0

    def save_checkpoint(self, 
                        cfg,
                        state,
                        is_best=False,
                        remove_module_from_keys=True,
                        model_name=""):
        if remove_module_from_keys:
            # remove 'module.' in state_dict's keys
            state_dict = state["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            state["state_dict"] = new_state_dict

        # save model
        epoch = state["epoch"]
        if not model_name:
            model_name = "model.pth.tar-" + str(epoch)

        # Make model dir if not exists
        mkdir_if_missing(osp.join(cfg.MODEL.OUTPUT_DIR, "model"))

        fpath = osp.join(cfg.MODEL.OUTPUT_DIR, "model", model_name)
        torch.save(state, fpath)
        logger.info(f"Checkpoint saved to {fpath}")

        # save current model name
        checkpoint_file = osp.join(cfg.MODEL.OUTPUT_DIR, "model", "checkpoint")
        checkpoint = open(checkpoint_file, "w+")
        checkpoint.write("{}\n".format(osp.basename(fpath)))
        logger.info(f"Checkpoint file containing model name '{osp.basename(fpath)}' is written to {checkpoint_file}")
        checkpoint.close()

        if is_best:
            best_fpath = osp.join(osp.dirname(fpath), "model-best.pth.tar")
            shutil.copy(fpath, best_fpath)
            logger.info('Best checkpoint saved to "{}"'.format(best_fpath))

def build_model(cfg):
    if cfg.MODEL.TYPE == "MyModel":
        logger.info(f"Loading my model...")
        return MyModel(cfg=cfg)
    elif cfg.MODEL.TYPE == "Baseline":
        logger.info(f"Loading baseline...")
        return Baseline(cfg=cfg)
    else:
        logger.error(f"Unknown model type: {cfg.MODEL.TYPE}")
        raise ValueError(f"Unknown model type: {cfg.MODEL.TYPE}")
