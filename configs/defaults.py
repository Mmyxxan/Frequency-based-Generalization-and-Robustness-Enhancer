# my_project/config.py
from yacs.config import CfgNode as CN


_C = CN()

# MODEL
_C.MODEL = CN()
_C.MODEL.TYPE = "MyModel"
_C.MODEL.NAME = "Fused_CNN_ResNet50_CLIP_ViT_512_Concat"
_C.MODEL.OUTPUT_DIR = f"output/{_C.MODEL.TYPE}/{_C.MODEL.NAME}"
_C.MODEL.MODEL_PATH = f"output/{_C.MODEL.TYPE}/{_C.MODEL.NAME}/model/model-best.pth.tar"
# BACKBONE
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.PRETRAINED = False
_C.MODEL.BACKBONE.FREEZE = True

# DATASET
_C.DATASET = CN()
_C.DATASET.NAME = "ProGAN"
_C.DATASET.DATA_DIR = "" # e.g., cnnspot/test/progan, cnnspot/train
_C.DATASET.NUM_CLASSES = 2 # for classifier to know about the dataset

# DATALOADER
_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 128
_C.DATALOADER.NUM_WORKERS = 4

# TRANSFORM
_C.TRANSFORM = CN()
# TRAIN
_C.TRANSFORM.TRANSFORMS = ("resize", "jpeg_compression", "to_tensor", "normalize")
# _C.TRANSFORM.TRANSFORMS = ("resize", "random_flip", "gaussian_blur", "jpeg_compression", "to_tensor", "normalize")
_C.TRANSFORM.NORMALIZE_BACKBONE = True # normalize will be done by each backbone's mean and std
_C.TRANSFORM.INTERPOLATION_MODE = "bilinear" # for resnet50, bilinear is often used, for clip, bicubic
_C.TRANSFORM.INPUT_SIZE = (224, 224)
_C.TRANSFORM.INPUT_MEAN = [0.485, 0.456, 0.406]
_C.TRANSFORM.INPUT_STD = [0.229, 0.224, 0.225]
# Gaussian blur
_C.TRANSFORM.GB_P = 0.5  # propability of applying this operation
_C.TRANSFORM.GB_K = 21  # kernel size (should be an odd number)
_C.TRANSFORM.GB_SIGMA = (0.1, 3.0)  # Uniform[0, 3]
# Jpeg compression
_C.TRANSFORM.JPEG_P = 0.5
_C.TRANSFORM.JPEG_QUALITY = (30, 99)
# TEST
_C.TRANSFORM.NO_TRANSFORM_TEST = False

# TRAINER
_C.TRAINER = CN()
_C.TRAINER.USE_CUDA = True
_C.TRAINER.IS_TRAIN = False

# EVALUATOR
_C.EVALUATOR = CN()
