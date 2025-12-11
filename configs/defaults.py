# my_project/config.py
from yacs.config import CfgNode as CN


_C = CN()

# MODEL
_C.MODEL = CN()
_C.MODEL.TYPE = "MyModel"
_C.MODEL.NAME = "Fused_CNN_ResNet50_CLIP_ViT_512_Concat"
_C.MODEL.OUTPUT_DIR = f"output/{_C.MODEL.TYPE}/{_C.MODEL.NAME}"
_C.MODEL.MODEL_DIR = f"output/{_C.MODEL.TYPE}/{_C.MODEL.NAME}/model"
_C.MODEL.MODEL_NAME = "model-best.pth.tar"
_C.MODEL.RESUME = True # default to resume training
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
# GAUSSIAN BLUR
_C.TRANSFORM.GB_P = 0.5  # propability of applying this operation
_C.TRANSFORM.GB_K = 21  # kernel size (should be an odd number)
_C.TRANSFORM.GB_SIGMA = (0.1, 3.0)  # Uniform[0, 3]
# JPEG COMPRESSION
_C.TRANSFORM.JPEG_P = 0.5
_C.TRANSFORM.JPEG_QUALITY = (30, 99)
# TEST
_C.TRANSFORM.NO_TRANSFORM_TEST = False

# TRAINER
_C.TRAINER = CN()
_C.TRAINER.TYPE = 0
_C.TRAINER.USE_CUDA = True
_C.TRAINER.IS_TRAIN = False
_C.TRAINER.NUM_EPOCHS = 1
_C.TRAINER.NO_TEST = False # val after epoch and test after train
_C.TRAINER.TEST_FINAL_MODEL = "best_val" # best_val or last_epoch
_C.TRAINER.CHECKPOINT_FREQ = 0
_C.TRAINER.PRINT_FREQ = 10
_C.TRAINER.SEED = -1
# JaFR
_C.TRAINER.JaFR = CN()
_C.TRAINER.JaFR.FREQ_BIAS_LAMBDA = 1e-3
_C.TRAINER.JaFR.TRACK_FREQ_BIAS_LOSS = False
_C.TRAINER.JaFR.EPS = 8.0/255
_C.TRAINER.JaFR.DELTA_TYPE_FOR_GRAD_BACKPROP = 'none' # ['none', 'random_uniform']
_C.TRAINER.JaFR.MAX_POW = 1
_C.TRAINER.JaFR.FREQ_BIAS_TEMPERATURE = 1
_C.TRAINER.JaFR.FREQ_BIAS_REDUCE_TYPE = 'sum' # ['sumlog', 'sum', 'product']
_C.TRAINER.JaFR.FREQ_BIAS_IGNORE_FIRST_BASIS = False
_C.TRAINER.JaFR.EPOCHS_WARMUP_BEFORE_FREQ_BIAS_REG = -1
# Optimization
_C.TRAINER.OPTIM = CN()
_C.TRAINER.OPTIM.NAME = "adam"
_C.TRAINER.OPTIM.LR = 0.0003
_C.TRAINER.OPTIM.WEIGHT_DECAY = 5e-4
_C.TRAINER.OPTIM.MOMENTUM = 0.9
_C.TRAINER.OPTIM.SGD_DAMPNING = 0
_C.TRAINER.OPTIM.SGD_NESTEROV = False
_C.TRAINER.OPTIM.RMSPROP_ALPHA = 0.99
# The following also apply to other
# adaptive optimizers like adamw
_C.TRAINER.OPTIM.ADAM_BETA1 = 0.9
_C.TRAINER.OPTIM.ADAM_BETA2 = 0.999
# STAGED_LR allows different layers to have
# different lr, e.g. pre-trained base layers
# can be assigned a smaller lr than the new
# classification layer
_C.TRAINER.OPTIM.STAGED_LR = False
_C.TRAINER.OPTIM.NEW_LAYERS = ()
_C.TRAINER.OPTIM.BASE_LR_MULT = 0.1
# Learning rate scheduler
_C.TRAINER.OPTIM.LR_SCHEDULER = "single_step"
# -1 or 0 means the stepsize is equal to max_epoch
_C.TRAINER.OPTIM.STEPSIZE = (-1, )
_C.TRAINER.OPTIM.GAMMA = 0.1
_C.TRAINER.OPTIM.MAX_EPOCH = _C.TRAINER.NUM_EPOCHS
# Set WARMUP_EPOCH larger than 0 to activate warmup training
_C.TRAINER.OPTIM.WARMUP_EPOCH = -1
# Either linear or constant
_C.TRAINER.OPTIM.WARMUP_TYPE = "linear"
# Constant learning rate when type=constant
_C.TRAINER.OPTIM.WARMUP_CONS_LR = 1e-5
# Minimum learning rate when type=linear
_C.TRAINER.OPTIM.WARMUP_MIN_LR = 1e-5
# Recount epoch for the next scheduler (last_epoch=-1)
# Otherwise last_epoch=warmup_epoch
_C.TRAINER.OPTIM.WARMUP_RECOUNT = True

# JaFR
_C.JaFR = CN()
_C.JaFR.VISUALIZE_ONLY = False
_C.JaFR.VISUALIZE_FOURIER_DATASET = False
_C.JaFR.VISUALIZE_JACOBIAN_MODEL = False

# EVALUATOR
_C.EVALUATOR = CN()
_C.EVALUATOR.COMPUTE_CONFUSION_MATRIX = False
