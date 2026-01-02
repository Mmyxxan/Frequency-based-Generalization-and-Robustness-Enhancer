# need refactor
import time
import datetime
from tqdm import tqdm
import os.path as osp
import json

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from timm.loss import JsdCrossEntropy
# from torchmetrics.image import TotalVariation
# https://lightning.ai/docs/torchmetrics/stable/image/total_variation.html
from torchvision import transforms
from torchvision.transforms import v2
# https://docs.pytorch.org/vision/main/generated/torchvision.transforms.AugMix.html
# Training details: https://arxiv.org/pdf/2103.16241

from models import build_model
from data import build_dataloader, build_transform
from utils import count_num_param, AverageMeter, MetricMeter
from evaluators import build_evaluator, compute_accuracy, accuracy
from .optimizer import build_optimizer
from .lr_scheduler import build_lr_scheduler

from models import Baseline
from utils import * 
from utils import logger

# Trainer inherits Abstract trainer for different training flows based on cfg.TRAINER.TYPE

class AbstractTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        # Set up device
        if self.cfg.TRAINER.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Build model
        self.model = build_model(self.cfg)
        self.model.to(self.device)
        logger.info(f"Number of params: {count_num_param(self.model, trainable_only=False):,}")
        logger.info(f"Number of trainable params: {count_num_param(self.model, trainable_only=True):,}")
        # Detect devices
        device_count = torch.cuda.device_count()
        if device_count > 1:
            logger.info(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)

        if self.cfg.TRAINER.IS_TRAIN:
            logger.info(f"Training {self.cfg.MODEL.NAME}")
            # If train, initialize best result
            self.best_result = -np.inf
            # If train, set up number of epochs by default
            self.start_epoch = 0
            self.last_epoch = self.cfg.TRAINER.NUM_EPOCHS
            # If train, build optimizer and lr_scheduler
            self.optimizer = build_optimizer(self.model, self.cfg.TRAINER.OPTIM)
            self.scheduler = build_lr_scheduler(self.optimizer, self.cfg.TRAINER.OPTIM)
            # If train, build train and val loader
            self.train_loader = build_dataloader(self.cfg, is_train=True, split="train")
            logger.info("Successfully build train loader!")
            if not self.cfg.TRAINER.NO_TEST:
                self.val_loader = build_dataloader(self.cfg, is_train=False, split="val")
                logger.info("Successfully build val loader!")
            else:
                logger.info("No test, no need to build val loader!")

        if not self.cfg.TRAINER.NO_TEST:
            # Build test loader
            self.test_loader = build_dataloader(self.cfg, is_train=False, split="test")
            logger.info("Successfully build test loader!")

            # Build evaluator
            self.evaluator = build_evaluator(self.cfg)
            logger.info("Successfully build evaluator!")
        else:
            logger.info("No test, no need to build test loader and evaluator!")

    def get_model(self):
        if isinstance(self.model, Baseline):
            return self.model.model.module if isinstance(self.model.model, nn.DataParallel) else self.model.model
        else:
            return self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
    def set_model_mode(self, mode="train"):
        if mode == "train":
            self.model.train()
        elif mode in ["val", "test"]:
            self.model.eval()
        else:
            logger.error(f"Unknown key {mode}")
            raise KeyError(f"Unknown key {mode}")

    def train(self):
        """Generic training loops."""
        self.set_model_mode(mode="train")

        self.before_train()
        for self.epoch in range(self.start_epoch, self.last_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        optimizer = getattr(self, "optimizer", None)
        scheduler = getattr(self, "scheduler", None)

        self.start_epoch = self.get_model().resume_or_load_checkpoint(
            self.cfg,
            optimizer,
            scheduler
        )

        if self.start_epoch >= self.cfg.TRAINER.NUM_EPOCHS:
            logger.error(f"Start epoch ({self.start_epoch}) is larger or equal to number of epochs ({self.cfg.TRAINER.NUM_EPOCHS})!")
            raise ValueError(f"Start epoch ({self.start_epoch}) is larger or equal to number of epochs ({self.cfg.TRAINER.NUM_EPOCHS})!")

        self.time_start = time.time()

    def before_epoch(self):
        pass

    def run_epoch(self):
        pass

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.last_epoch
        do_test = not self.cfg.TRAINER.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAINER.CHECKPOINT_FREQ == 0
            if self.cfg.TRAINER.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TRAINER.TEST_FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.get_model().save_checkpoint(
                    cfg=self.cfg,
                    state={
                        "state_dict": self.model.state_dict(),
                        "epoch": self.epoch + 1,
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "val_result": self.best_result
                    },
                    is_best=True,
                )

        if meet_checkpoint_freq or last_epoch:
            self.get_model().save_checkpoint(
                cfg=self.cfg,
                state={
                    "state_dict": self.model.state_dict(),
                    "epoch": self.epoch + 1,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "val_result": None
                },
                is_best=False,
            )

    def after_train(self):
        logger.info("Finish training!")

        if not self.cfg.TRAINER.NO_TEST:
            if self.cfg.TRAINER.TEST_FINAL_MODEL == "best_val":
                logger.info("Deploy the model with the best val performance")
                self.get_model().load_best_model(f"{self.cfg.MODEL.OUTPUT_DIR}/model")
            else:
                logger.info("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logger.info(f"Elapsed: {elapsed}")
    
    @torch.no_grad()
    def test(self, split="test"):
        """A generic testing pipeline."""
        self.set_model_mode(split)
        self.evaluator.reset()

        if split == "test":
            data_loader = self.test_loader
        elif split == "val":
            data_loader = self.val_loader

        logger.info(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logger.info(f"Elapsed: {elapsed}")

        return list(results.values())[0]
    
    def model_inference(self, input):
        return self.model(input)

    def parse_batch_test(self, batch):
        input = batch[0]
        label = batch[1]

        if isinstance(input, list):
            input = [x.to(self.device) for x in input]
        else:
            input = input.to(self.device)

        label = label.to(self.device)

        return input, label

    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]
    
    def update_lr(self):
        if self.scheduler is not None:
            self.scheduler.step()
    
    def model_zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            logger.error("Loss is infinite or NaN!")
            raise FloatingPointError("Loss is infinite or NaN!")

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self):
        if self.optimizer is not None:
            self.optimizer.step()
    
    def model_backward_and_update(self, loss):
        self.model_zero_grad()
        self.model_backward(loss)
        self.model_update()

    def inspect_weights(self):
        backbone = self.model.backbone
        
        if not hasattr(backbone, "backbone_list"):
            logger.error("Backbone is not fused; no extractors to inspect.")
            raise KeyError("Backbone is not fused; no extractors to inspect.")

        num_extractors = len(backbone.backbone_list)
        logger.info(f"Found {num_extractors} extractors!")
        project_dim = backbone.projections[0].out_features  

        classifier = self.model.classifier           # linear layer
        W = classifier.weight.detach()               # shape: (C, total_dim)

        contributions = []
        start = 0

        for idx, extractor_cls in enumerate(backbone.backbone_list):
            end = start + project_dim
            W_block = W[:, start:end]                 # slice for this extractor

            contrib = W_block.abs().sum().item()      # L1 importance

            contributions.append((extractor_cls.__name__, contrib))
            start = end

        # normalize
        total = sum(c for _, c in contributions)
        contributions_pct = [(name, c, c / total * 100) for name, c in contributions]

        logger.info("=== Extractor Contribution Analysis ===")
        for name, raw, pct in contributions_pct:
            logger.info(f"{name:<25} | Raw: {raw:10.4f} | {pct:6.2f}%")

        return contributions_pct

class StandardTrainer(AbstractTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        logger.info("Successfully build StandardTrainer!")

    def run_epoch(self):
        self.set_model_mode("train")

        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader)

        end = time.time()

        for self.batch_idx, batch in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAINER.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAINER.PRINT_FREQ

            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.last_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.last_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                logger.info(" ".join(info))

            end = time.time()

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        output = self.model(input)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch[0]
        label = batch[1]

        if isinstance(input, list):
            input = [x.to(self.device) for x in input]
        else:
            input = input.to(self.device)

        label = label.to(self.device)

        return input, label
        
class BaselineTester(AbstractTrainer):
    def __init__(self, cfg):
        self.cfg = cfg
        # Set up device
        if self.cfg.TRAINER.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Build model
        self.model = build_model(self.cfg)
        self.model.model.to(self.device)
        logger.info(f"Number of params: {count_num_param(self.model.model, trainable_only=False):,}")

        # Build test loader
        self.test_loader = build_dataloader(self.cfg, is_train=False, split="test")
        logger.info("Successfully build test loader!")

        # Build evaluator
        self.evaluator = build_evaluator(self.cfg)
        logger.info("Successfully build evaluator!")

        logger.info("Successfully build BaselineTester!")

    def set_model_mode(self, mode="test"):
        if mode == "test":
            self.model.model.eval()
        else:
            logger.error(f"Unknown key {mode}")
            raise KeyError(f"Unknown key {mode}")

    def train(self):
        pass

    def before_train(self):
        self.model.load_checkpoint(self.cfg)

        # Detect devices
        device_count = torch.cuda.device_count()
        if device_count > 1:
            logger.info(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model.model = nn.DataParallel(self.model.model)

        self.time_start = time.time()

    def before_epoch(self):
        pass

    def run_epoch(self):
        pass

    def after_epoch(self):
        pass

    def after_train(self):
        pass
    
    @torch.no_grad()
    def test(self, split="test"):
        split = "test" # only test in this trainer

        self.set_model_mode(split)
        self.evaluator.reset()

        data_loader = self.test_loader

        logger.info(f"Evaluate on the *test* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logger.info(f"Elapsed: {elapsed}")

        return list(results.values())[0]
    
    def model_inference(self, input):
        return self.model.forward(input)

    def parse_batch_test(self, batch):
        input = batch[0]
        label = batch[1]

        if isinstance(input, list):
            input = [x.to(self.device) for x in input]
        else:
            input = input.to(self.device)

        label = label.to(self.device)

        return input, label

    def get_current_lr(self):
        pass
    
    def update_lr(self):
        pass
    
    def model_zero_grad(self):
        pass

    def detect_anomaly(self, loss):
        pass

    def model_backward(self, loss):
        pass

    def model_update(self):
        pass
    
    def model_backward_and_update(self, loss):
        pass

    def inspect_weights(self):
        pass
    
    def forward_backward(self, batch):
        pass

    def parse_batch_train(self, batch):
        pass

class JaFRTrainer(AbstractTrainer):
    def __init__(self, cfg):
        self.cfg = cfg

        # Set up device
        if self.cfg.TRAINER.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Build model
        if not self.cfg.JaFR.VISUALIZE_ONLY or self.cfg.JaFR.VISUALIZE_JACOBIAN_MODEL:
            self.model = build_model(self.cfg)
            if isinstance(self.model, Baseline):
                self.model.model.to(self.device)
                logger.info(f"Number of params: {count_num_param(self.model.model, trainable_only=False):,}")
                logger.info(f"Number of trainable params: {count_num_param(self.model.model, trainable_only=True):,}")
            else:
                self.model.to(self.device)
                logger.info(f"Number of params: {count_num_param(self.model, trainable_only=False):,}")
                logger.info(f"Number of trainable params: {count_num_param(self.model, trainable_only=True):,}")
        
        # Detect devices
        device_count = torch.cuda.device_count()
        if device_count > 1:
            logger.info(f"Detected {device_count} GPUs (use nn.DataParallel)")
            if isinstance(self.model, Baseline):
                self.model.model = nn.DataParallel(self.model.model)
            else:
                self.model = nn.DataParallel(self.model)
            
        if self.cfg.TRAINER.IS_TRAIN:
            logger.info(f"Training {self.cfg.MODEL.NAME}")
            # If train, initialize best result
            self.best_result = -np.inf # best
            # If train, set up number of epochs by default
            self.start_epoch = 0
            self.last_epoch = self.cfg.TRAINER.NUM_EPOCHS
            # If train, build optimizer and lr_scheduler
            self.optimizer = build_optimizer(self.model, self.cfg.TRAINER.OPTIM)
            self.scheduler = build_lr_scheduler(self.optimizer, self.cfg.TRAINER.OPTIM)
            # If train, build train and val loader
            self.train_loader = build_dataloader(self.cfg, is_train=True, split="train")
            logger.info("Successfully build train loader!")
            if not self.cfg.TRAINER.NO_TEST:
                self.val_loader = build_dataloader(self.cfg, is_train=False, split="val")
                logger.info("Successfully build val loader!")
            else:
                logger.info("No test, no need to build val loader!")

        # Build test loader
        if not self.cfg.TRAINER.NO_TEST or self.cfg.JaFR.VISUALIZE_JACOBIAN_MODEL:
            self.test_loader = build_dataloader(self.cfg, is_train=False, split="test")
            logger.info("Successfully build test loader!")
        else:
            logger.info("No test and no visualize fourier of dataset, no need to build test loader!")

        # Build evaluator
        if not self.cfg.TRAINER.NO_TEST:
            self.evaluator = build_evaluator(self.cfg)
            logger.info("Successfully build evaluator!")
        else:
            logger.info("No test, no need to build evaluator!")
        
        # Build dataloader for visualization
        if self.cfg.JaFR.VISUALIZE_FOURIER_DATASET:
            self.visualize_dataloader = build_dataloader(self.cfg, is_train=False, split="test", is_visualize=True)

    def set_model_mode(self, mode):
        if mode == "train":
            if isinstance(self.model, Baseline):
                self.model.model.train()
            else:
                self.model.train()
        elif mode in ["val", "test"]:
            if isinstance(self.model, Baseline):
                self.model.model.eval()
            else:
                self.model.eval()
        else:
            logger.error(f"Unknown key {mode}")
            raise KeyError(f"Unknown key {mode}")
        
    # keep generic train function as abstract trainer

    def before_train(self):
        if isinstance(self.model, Baseline):
            self.model.load_checkpoint(self.cfg)
        else:
            optimizer = getattr(self, "optimizer", None)
            scheduler = getattr(self, "scheduler", None)

            self.start_epoch = self.get_model().resume_or_load_checkpoint(
                self.cfg,
                optimizer,
                scheduler
            )

            if self.cfg.TRAINER.IS_TRAIN and self.start_epoch >= self.cfg.TRAINER.NUM_EPOCHS:
                logger.error(f"Start epoch ({self.start_epoch}) is larger or equal to number of epochs ({self.cfg.TRAINER.NUM_EPOCHS})!")
                raise ValueError(f"Start epoch ({self.start_epoch}) is larger or equal to number of epochs ({self.cfg.TRAINER.NUM_EPOCHS})!")

        self.time_start = time.time()

    # keep generic before_epoch function as abstract trainer

    def run_epoch(self):
        self.set_model_mode("train")

        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader)

        end = time.time()

        for self.batch_idx, batch in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            if not loss_summary:
                break
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAINER.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAINER.PRINT_FREQ

            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.last_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.last_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                logger.info(" ".join(info))

            end = time.time()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.last_epoch
        do_test = not self.cfg.TRAINER.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAINER.CHECKPOINT_FREQ == 0
            if self.cfg.TRAINER.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TRAINER.TEST_FINAL_MODEL == "best_val":
            curr_result = self.test(split="val") # take accuracy, not the test loss for representative
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.get_model().save_checkpoint(
                    cfg=self.cfg,
                    state={
                        "state_dict": self.model.state_dict(),
                        "epoch": self.epoch + 1,
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "val_result": self.best_result
                    },
                    is_best=True,
                )

        if meet_checkpoint_freq or last_epoch:
            self.get_model().save_checkpoint(
                cfg=self.cfg,
                state={
                    "state_dict": self.model.state_dict(),
                    "epoch": self.epoch + 1,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "val_result": None
                },
                is_best=False,
            )

    # keep generic after_train function as abstract trainer

    @torch.no_grad()
    def test(self, split="test"):
        self.set_model_mode(split)
        self.evaluator.reset()

        if split == "test":
            data_loader = self.test_loader
        elif split == "val":
            data_loader = self.val_loader

        logger.info(f"Evaluate on the *{split}* set")

        # total_loss = 0.0
        # n_batches = 0

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            # loss = self.compute_loss(input=input, output=output, label=label)[0]

            # total_loss += loss.item() # add scalar
            # n_batches += 1

            self.evaluator.process(output, label)

        # Compute average test loss
        # avg_loss = total_loss / n_batches

        results = self.evaluator.evaluate()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logger.info(f"Elapsed: {elapsed}")

        # return list(results.values())[0], avg_loss
        return list(results.values())[0]
    
    def model_inference(self, input):
        if isinstance(self.model, Baseline):
            return self.model.forward(input)
        else:
            return self.model(input)
    
    # keep generic parse_batch_test, get_current_lr, update_lr, model_zero_grad, detect_anomaly, model_backward, model_update, model_backward_and_update, inspect_weights functions as abstract trainer

    def compute_loss(self, input, output, label):
        if isinstance(self.model, Baseline):
            model = self.model.model
        else:
            model = self.model
        
        loss = F.cross_entropy(output, label)

        # Only applicable to square training images like ProGAN
        if self.cfg.TRAINER.USE_CUDA:
            low_freq_bias_reg = torch.zeros(1).cuda()[0]
        else: 
            low_freq_bias_reg = torch.zeros(1)[0]

        if self.cfg.TRAINER.USE_CUDA:
            high_freq_bias_reg = torch.zeros(1).cuda()[0]
        else: 
            high_freq_bias_reg = torch.zeros(1)[0]

        if self.cfg.TRAINER.JaFR.FREQ_BIAS_LAMBDA != 0.0 or self.cfg.TRAINER.JaFR.TRACK_FREQ_BIAS_LOSS:
            if self.cfg.TRAINER.JaFR.FREQ_BIAS_LAMBDA == 0.0:                
                grads_for_backprop = get_grad_extractor(model, input, label, None, self.cfg.TRAINER.JaFR.EPS, None, 
                                    delta_init=self.cfg.TRAINER.JaFR.DELTA_TYPE_FOR_GRAD_BACKPROP, backprop=False, cuda=self.cfg.TRAINER.USE_CUDA)
                grads_for_backprop = grads_for_backprop.detach()
            else:
                grads_for_backprop = get_grad_extractor(model, input, label, None, self.cfg.TRAINER.JaFR.EPS, None, 
                                    delta_init=self.cfg.TRAINER.JaFR.DELTA_TYPE_FOR_GRAD_BACKPROP, backprop=True, cuda=self.cfg.TRAINER.USE_CUDA)

            grads_to_reg_freq = grads_for_backprop[:]

            # grad_to_low_freq_reg = grads_to_reg_freq[0]
            # grad_to_high_freq_reg = grads_to_reg_freq[1]
            grad_to_low_freq_reg = grads_to_reg_freq[0].detach()
            grad_to_high_freq_reg = grads_to_reg_freq[1].detach()

            grad_low_freq_bias_value = self.compute_grad_low_freq_bias_value(grad_to_low_freq_reg)
            grad_high_freq_bias_value = -1 * self.compute_grad_low_freq_bias_value(grad_to_high_freq_reg)
                            
            low_freq_bias_reg += self.cfg.TRAINER.JaFR.FREQ_BIAS_LAMBDA * -1 * grad_low_freq_bias_value
            high_freq_bias_reg += self.cfg.TRAINER.JaFR.FREQ_BIAS_LAMBDA * -1 * grad_high_freq_bias_value

            if self.cfg.TRAINER.JaFR.FREQ_BIAS_LAMBDA != 0.0 and self.epoch > self.cfg.TRAINER.JaFR.EPOCHS_WARMUP_BEFORE_FREQ_BIAS_REG:
                reg = low_freq_bias_reg + high_freq_bias_reg

                if torch.isfinite(reg).all():
                    # reg = torch.clamp(reg, min=-0.1, max=0.1)
                    loss = loss + reg
                else:
                    logger.warning(
                        f"Non-finite freq reg skipped: "
                        f"low={low_freq_bias_reg.item()}, "
                        f"high={high_freq_bias_reg.item()}"
                    )
                # design loss function to push two extractors to different frequency bias (the larger B_low_1 - B_low_2, the better)
            elif self.cfg.TRAINER.JaFR.FREQ_BIAS_LAMBDA == 0.0 and self.cfg.TRAINER.JaFR.TRACK_FREQ_BIAS_LOSS:
                low_freq_bias_reg += -1 * grad_low_freq_bias_value
                high_freq_bias_reg += -1 * grad_high_freq_bias_value

            # sample01_low_freq_norm = chmean_grad_freq_norm[:1+chmean_grad_freq_norm.shape[0]//2][0,1]
            # sample10_low_freq_norm = chmean_grad_freq_norm[:1+chmean_grad_freq_norm.shape[0]//2][1,0]

            # if self.cfg.DATASET.NAME == "CNNSpot":
            #     sample_high_freq_norm = chmean_grad_freq_norm[:1+chmean_grad_freq_norm.shape[0]//2][128,128]
            # else:
            #     logger.error(f"Unapplicable dataset {self.cfg.DATASET.NAME}")
            #     raise ValueError(f"Unapplicable dataset {self.cfg.DATASET.NAME}")

            # logger.info('01_low_freq_norm: {}'.format(sample01_low_freq_norm.item()))
            # logger.info('10_low_freq_norm: {}'.format(sample10_low_freq_norm.item()))
            # logger.info('high_freq_norm: {}'.format(sample_high_freq_norm.item()))

        logger.debug('loss {}, low_freq_bias_reg {}, high_freq_bias_reg {}'.format(loss.item(), low_freq_bias_reg.item(), high_freq_bias_reg.item()))
        # high_freq_bias_reg is -inf without warmup
        # RuntimeError: derivative for aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented

        return loss, low_freq_bias_reg, high_freq_bias_reg
    
    def compute_grad_low_freq_bias_value(self, grad_to_reg_freq):
        chmean_grad_freq_norm = compute_fourier_map(grad_to_reg_freq)

        grad_low_freq_bias_value = compute_low_freq_bias(chmean_grad_freq_norm[:1+chmean_grad_freq_norm.shape[0]//2], 
                                        max_pow=self.cfg.TRAINER.JaFR.MAX_POW, min_pow=-1*self.cfg.TRAINER.JaFR.MAX_POW, temperature=self.cfg.TRAINER.JaFR.FREQ_BIAS_TEMPERATURE, reduce_type=self.cfg.TRAINER.JaFR.FREQ_BIAS_REDUCE_TYPE, ignore_first_basis=self.cfg.TRAINER.JaFR.FREQ_BIAS_IGNORE_FIRST_BASIS, cuda=self.cfg.TRAINER.USE_CUDA)

        return grad_low_freq_bias_value
        
    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        output = self.model(input)
        loss, low_freq_bias_reg, high_freq_bias_reg = self.compute_loss(input=input, output=output, label=label)
        try:
            self.model_backward_and_update(loss)
        except Exception:
            logger.error("NaN or infinity during backward, saving checkpoint")
            self.get_model().save_checkpoint(
                cfg=self.cfg,
                state={
                    "state_dict": self.model.state_dict(),
                    "epoch": self.epoch + 1,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "val_result": None
                },
                is_best=False,
            )
            return None

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            "low_freq_bias_reg": low_freq_bias_reg.item(),
            "high_freq_bias_reg": high_freq_bias_reg.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch[0]
        label = batch[1]

        if isinstance(input, list):
            input = [x.to(self.device) for x in input]
            # self.delta = [torch.zeros_like(x, requires_grad=True) for x in input] # not carry out adversarial training
            # input = [x + d for x, d in zip(input, self.delta)]
        else:
            input = input.to(self.device)
            # self.delta = torch.zeros_like(input, requires_grad=True)
            # input = input + self.delta

        label = label.to(self.device)

        return input, label

    def visualize_fourier_dataset(self):
        time_start = time.time()

        logger.info("Running Fourier frequency bias analysis...")

        analysis_output_dir = f"output/JaFR/{self.cfg.DATASET.NAME}/analysis"
        mkdir_if_missing(analysis_output_dir)

        cor_diff_low_freq_bias_value = analyze_corruption_fourier_and_freq_bias(self.visualize_dataloader, self.visualize_dataloader, analysis_output_dir=analysis_output_dir, output_dir_suffix="", cuda=self.cfg.TRAINER.USE_CUDA)

        logger.info(f"Fourier analysis result: {cor_diff_low_freq_bias_value}")

        json_path = os.path.join(analysis_output_dir, "analysis_results.json")
        with open(json_path, 'w') as f:
            json.dump({"fourier_bias": cor_diff_low_freq_bias_value.cpu().numpy().tolist()}, f)
        logger.info(f"Saved results to {json_path}.")
    
        time_elapsed = time.time() - time_start
        time_elapsed = str(datetime.timedelta(seconds=time_elapsed))
        logger.info(f"Elapsed: {time_elapsed}")

    def visualize_jacobian_model(self):
        time_start = time.time()

        logger.info("Running Jacobian analysis for model...")

        if isinstance(self.model, Baseline):
            model = self.model.model
        else:
            model = self.model

        analysis_output_dir = f"output/JaFR/{self.cfg.MODEL.TYPE}_{self.cfg.MODEL.NAME}_{self.cfg.DATASET.NAME}/analysis"
        mkdir_if_missing(analysis_output_dir)

        grad_low_freq_bias, img_low_freq_bias = analyze_save_ig(self.test_loader, model, None, self.cfg.TRAINER.JaFR.EPS, None, model_output_dir=analysis_output_dir,
                                                        max_pow=self.cfg.TRAINER.JaFR.MAX_POW, min_pow=-1*self.cfg.TRAINER.JaFR.MAX_POW, temperature=self.cfg.TRAINER.JaFR.FREQ_BIAS_TEMPERATURE, cuda=self.cfg.TRAINER.USE_CUDA)
        randinit_grad_low_freq_bias, _ = analyze_save_ig(self.test_loader, model, None, self.cfg.TRAINER.JaFR.EPS, None, model_output_dir=analysis_output_dir, delta_init='random_uniform', output_dir_suffix='_randinitgrad',
                                            max_pow=self.cfg.TRAINER.JaFR.MAX_POW, min_pow=-1*self.cfg.TRAINER.JaFR.MAX_POW, temperature=self.cfg.TRAINER.JaFR.FREQ_BIAS_TEMPERATURE, cuda=self.cfg.TRAINER.USE_CUDA)
        
        logger.info('grad_low_freq_bias {}, img_low_freq_bias {}, randinit_grad_low_freq_bias {}'.format(grad_low_freq_bias, img_low_freq_bias, randinit_grad_low_freq_bias))
        eval_results = 'grad_low_freq_bias {}, img_low_freq_bias {}, randinit_grad_low_freq_bias {}'.format(grad_low_freq_bias, img_low_freq_bias, randinit_grad_low_freq_bias)

        output_eval_file = os.path.join(analysis_output_dir, 'eval_results.txt')
        with open(output_eval_file, 'w') as f:
            f.write(eval_results)
        logger.info(f"Saved results to {output_eval_file}.")

        time_elapsed = time.time() - time_start
        time_elapsed = str(datetime.timedelta(seconds=time_elapsed))
        logger.info(f"Elapsed: {time_elapsed}")

class RoHLTrainer(AbstractTrainer):
    def __init__(self, cfg):
        self.cfg = cfg

        # Set up device
        if self.cfg.TRAINER.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Build model
        self.model = build_model(self.cfg)
        self.model.to(self.device)
        logger.info(f"Number of params: {count_num_param(self.model, trainable_only=False):,}")
        logger.info(f"Number of trainable params: {count_num_param(self.model, trainable_only=True):,}")
        
        # Detect devices
        device_count = torch.cuda.device_count()
        if device_count > 1:
            logger.info(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)
            
        if self.cfg.TRAINER.IS_TRAIN:
            logger.info(f"Training {self.cfg.MODEL.NAME}")
            # If train, build optimizer and lr_scheduler
            self.optimizer = build_optimizer(self.model, self.cfg.TRAINER.OPTIM)
            self.scheduler = build_lr_scheduler(self.optimizer, self.cfg.TRAINER.OPTIM)
            # If train, build train and val loader
            # Build two train loaders, one for high-frequency bias and one for low-frequency bias
            # First, from original transform, make high-frequency bias transform and low-frequency bias transform
            # Assume that original transforms include "resize", "to_tensor", "normalize" in that order
            original_transform = build_transform(self.cfg, is_train=True)
            logger.info("Successfully build original transform!")
            high_freq_transform, low_freq_transform, augmix_transform = self.build_customized_transform(original_transform.transforms)
            
            # Notes:
            # AugMix pretrained ResNet50 is available
            # Write load pretrained AM, AMDA ResNet50 for my averaging model
            # The low-frequency expert was obtained by finetuning the publicly available AMDA model with contrast augmentation.
            # RoHL (AMDATV-ftGauss, AMDA-ftCont)

            # Then, build dataloader
            self.train_high_freq_loader = build_dataloader(self.cfg, is_train=True, split="train", transform=(high_freq_transform, original_transform))
            self.train_low_freq_loader = build_dataloader(self.cfg, is_train=True, split="train", transform=(low_freq_transform, original_transform))
            self.train_augmix_loader = build_dataloader(self.cfg, is_train=True, split="train", transform=(augmix_transform, original_transform)) # both high and low frequency corruptions for augmentation
            logger.info("Successfully build train loaders: high_freq, low_freq and augmix!")
            if not self.cfg.TRAINER.NO_TEST:
                self.val_loader = build_dataloader(self.cfg, is_train=False, split="val")
                logger.info("Successfully build val loader!")
            else:
                logger.info("No test, no need to build val loader!")

        # Build test loader
        if not self.cfg.TRAINER.NO_TEST:
            self.test_loader = build_dataloader(self.cfg, is_train=False, split="test")
            logger.info("Successfully build test loader!")
        else:
            logger.info("No test and no visualize fourier of dataset, no need to build test loader!")

        # Build evaluator
        if not self.cfg.TRAINER.NO_TEST:
            self.evaluator = build_evaluator(self.cfg)
            logger.info("Successfully build evaluator!")
        else:
            logger.info("No test, no need to build evaluator!")

    def build_customized_transform(self, original_transform):
        logger.info("Building customized transform...")
        logger.info("AUGMIX TRANSFORM...")
        augmix_transform = []
        for i, tf in original_transform:
            if i == 2:
                augmix_transform += [transforms.AugMix()] # AM
            augmix_transform.append(tf)
            logger.info(f"+ {augmix_transform[-1]}")
        augmix_transform = transforms.Compose(augmix_transform)

        logger.info("HIGH FREQUENCY TRANSFORM...")
        high_freq_transform = []
        for i, tf in original_transform:
            if i == 2:
                high_freq_transform += [transforms.AugMix(), transforms.GaussianBlur(3)] # AM_{TV}-ft_{Gauss}
            high_freq_transform.append(tf)
            if i == 2:
                high_freq_transform += [v2.GaussianNoise(mean=0, sigma=0.08)]
            logger.info(f"+ {high_freq_transform[-1]}")
        # We finetuned both AM and AMTV models with these HF augmentation operations. 
        high_freq_transform = transforms.Compose(high_freq_transform)

        logger.info("LOW FREQUENCY TRANSFORM...")
        low_freq_transform = []
        for i, tf in original_transform:
            if i == 2:
                low_freq_transform += [transforms.ColorJitter(contrast=0.4)] # AM-ft_{Cont}
            low_freq_transform.append(tf)
            logger.info(f"+ {low_freq_transform[-1]}")
        low_freq_transform = transforms.Compose(low_freq_transform)

        # Build actual mixed transform
        # transform = []
        # for i, tf in original_transform:
        #     if i == 2:
        #         gb_k, gb_p, gb_sigma = self.cfg.TRANSFORM.GB_K, self.cfg.TRANSFORM.GB_P, self.cfg.TRANSFORM.GB_SIGMA
        #         transform += [transforms.RandomApply([transforms.GaussianBlur(gb_k, gb_sigma)], p=gb_p)]

        #         jpeg_p, jpeg_quality = self.cfg.TRANSFORM.JPEG_P, self.cfg.TRANSFORM.JPEG_QUALITY
        #         transform += [transforms.RandomApply([v2.JPEG(quality=jpeg_quality)], p=jpeg_p)]
        #     transform.append(tf)
        # transform = transforms.Compose(transform)
        return high_freq_transform, low_freq_transform, augmix_transform
        
    def set_model_mode(self, mode):
        # There should be more model modes than train, val, test
        # In this trainer, model can be trained for high freq, low freq, adaptive average weights
        assert mode in ["train_augmix", "train_0", "train_1", "train_adaptive", "test_augmix", "test_0", "test_1", "test_adaptive"]

        self.model.set_model_mode(mode)
        
        if "train_" in mode:
            self.model.model.train()
        else:
            self.model.model.eval()
        
    def train(self):
        # Train AM first
        if self.cfg.RoHL.STAGE == 0:
            logger.info(f"Phase {self.cfg.RoHL.STAGE} of training RoHL model: training Augmix model!")
            self.set_model_mode(mode="train_augmix")
            self.before_train()
            for self.epoch in range(self.start_epoch, self.last_epoch):
                self.before_epoch()
                self.run_epoch("train_augmix")
                self.after_epoch("test_augmix") # only one branch of model
            self.after_train("test_augmix")

        if self.cfg.RoHL.STAGE == 1:
            logger.info(f"Phase {self.cfg.RoHL.STAGE} of training RoHL model: training the first ResNet50 towards low-frequency bias!")
            self.set_model_mode(mode="train_0")
            self.before_train()
            for self.epoch in range(self.start_epoch, self.last_epoch):
                self.before_epoch()
                self.run_epoch("train_0")
                self.after_epoch("test_0") # only one branch of model
            self.after_train("test_0")

        if self.cfg.RoHL.STAGE == 2:
            logger.info(f"Phase {self.cfg.RoHL.STAGE} of training RoHL model: training the second ResNet50 towards high-frequency bias!")
            self.set_model_mode(mode="train_1")
            self.before_train()
            for self.epoch in range(self.start_epoch, self.last_epoch):
                self.before_epoch()
                self.run_epoch("train_1")
                self.after_epoch("test_1") # two branches, fixed average
            self.after_train("test_1")

        # self.set_model_mode(mode="train_adaptive")
        # self.before_train()
        # for self.epoch in range(self.start_epoch, self.last_epoch):
        #     self.before_epoch()
        #     self.run_epoch("train_adaptive")
        #     self.after_epoch("test_adaptive") # two branches, adaptive average
        # self.after_train("test_adaptive")

    def before_train(self):
        optimizer = getattr(self, "optimizer", None)
        scheduler = getattr(self, "scheduler", None)

        # If train, initialize best result
        self.best_result = -np.inf # best
        # If train, set up number of epochs by default
        self.last_epoch = self.cfg.TRAINER.NUM_EPOCHS
        self.start_epoch = self.model.module.resume_or_load_checkpoint(
            self.cfg,
            optimizer,
            scheduler
        )

        if self.cfg.TRAINER.IS_TRAIN and self.start_epoch >= self.cfg.TRAINER.NUM_EPOCHS:
            logger.error(f"Start epoch ({self.start_epoch}) is larger or equal to number of epochs ({self.cfg.TRAINER.NUM_EPOCHS})!")
            raise ValueError(f"Start epoch ({self.start_epoch}) is larger or equal to number of epochs ({self.cfg.TRAINER.NUM_EPOCHS})!")

        self.time_start = time.time()

    # keep generic before_epoch function as abstract trainer

    def run_epoch(self, mode):
        self.set_model_mode(mode)
        if mode == "train_augmix":
            train_loader = self.train_augmix_loader
        elif mode == "train_0":
            train_loader = self.train_low_freq_loader
        elif mode == "train_1":
            train_loader = self.train_high_freq_loader
        elif mode == "train_adaptive":
            train_loader = self.train_loader

        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(train_loader)

        end = time.time()

        for self.batch_idx, batch in enumerate(train_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAINER.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAINER.PRINT_FREQ

            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.last_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.last_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                logger.info(" ".join(info))

            end = time.time()

    def load_duplicate_weights(self):
        self.model.load_duplicate_weights()
            
    def after_epoch(self, mode):
        last_epoch = (self.epoch + 1) == self.last_epoch
        do_test = not self.cfg.TRAINER.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAINER.CHECKPOINT_FREQ == 0
            if self.cfg.TRAINER.CHECKPOINT_FREQ > 0 else False
        )

        if mode == "test_augmix":
            logger.info("Copying AugMix weights to the other ResNet50...")
            self.load_duplicate_weights()

        if do_test and self.cfg.TRAINER.TEST_FINAL_MODEL == "best_val":
            curr_result = self.test(split="val", mode=mode)
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.get_model().save_checkpoint(
                    cfg=self.cfg,
                    state={
                        "state_dict": self.model.state_dict(),
                        "epoch": self.epoch + 1,
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "val_result": self.best_result
                    },
                    is_best=True,
                )

        if meet_checkpoint_freq or last_epoch:
            self.get_model().save_checkpoint(
                cfg=self.cfg,
                state={
                    "state_dict": self.model.state_dict(),
                    "epoch": self.epoch + 1,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "val_result": None
                },
                is_best=False,
            )

    def after_train(self, mode):
        logger.info("Finish training!")

        if not self.cfg.TRAINER.NO_TEST:
            if self.cfg.TRAINER.TEST_FINAL_MODEL == "best_val":
                logger.info("Deploy the model with the best val performance")
                self.get_model().load_best_model(f"{self.cfg.MODEL.OUTPUT_DIR}/model")
            else:
                logger.info("Deploy the last-epoch model")
            self.test(mode=mode)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logger.info(f"Elapsed: {elapsed}")
    
    @torch.no_grad()
    def test(self, mode, split="test"):
        """A generic testing pipeline."""
        self.set_model_mode(mode)
        self.evaluator.reset()

        if split == "test":
            data_loader = self.test_loader
        elif split == "val":
            data_loader = self.val_loader

        logger.info(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logger.info(f"Elapsed: {elapsed}")

        return list(results.values())[0]

    # keep generic parse_batch_test, get_current_lr, update_lr, model_zero_grad, detect_anomaly, model_backward, model_update, model_backward_and_update, inspect_weights functions as abstract trainer
    
    # keep order of functions inside trainer class
    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        if self.cfg.RoHL.USE_JSD:
            # Compute JSD loss https://github.com/google-research/augmix/blob/master/imagenet.py#L240
            # We employ AugMix data augmentation together with the JSD consistency loss and the default hyperparameters [18].
            images = input
            images_all = torch.cat(images, 0)
            targets = label
            logits_all = self.model(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
            # Cross-entropy is only computed on clean images
            loss = F.cross_entropy(logits_clean, targets)

            p_clean, p_aug1, p_aug2 = F.softmax(
                logits_clean, dim=1), F.softmax(
                    logits_aug1, dim=1), F.softmax(
                        logits_aug2, dim=1)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        else:
            output = self.model(input)
            loss = F.cross_entropy(output, label)

        # Should add Total Variation minimization loss

        self.model_backward_and_update(loss)

        if self.cfg.RoHL.USE_JSD:
            acc1, acc5 = accuracy(logits_clean, targets, topk=(1, 5))
            loss_summary = {
                "loss": loss.item(),
                "acc": acc1,
            }
        else:
            loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(output, label)[0].item(),
            }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch[0]
        label = batch[1]

        # Inspect input shape
        # logger.debug(input.shape)

        if isinstance(input, list):
            input = [x.to(self.device) for x in input]
        else:
            input = input.to(self.device)

        label = label.to(self.device)

        return input, label
