import time
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from models import build_model
from data import build_dataloader
from utils import count_num_param, AverageMeter, MetricMeter
from evaluators import build_evaluator, compute_accuracy
from .optimizer import build_optimizer
from .lr_scheduler import build_lr_scheduler

import logging
logger = logging.getLogger(__name__)

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
            # If train, initialize best result
            self.best_result = -np.inf
            # If train, set up number of epochs by default
            self.start_epoch = 0
            self.last_epoch = self.cfg.TRAINER.NUM_EPOCHS
            # If train, build optimizer and lr_scheduler
            self.optimizer = build_optimizer(self.model, self.cfg.OPTIM)
            self.scheduler = build_lr_scheduler(self.optimizer, self.cfg.OPTIM)
            # If train, build train and val loader
            self.train_loader = build_dataloader(self.cfg, is_train=True, split="train")
            self.val_loader = build_dataloader(self.cfg, is_train=False, split="val")

        # Build test loader
        self.test_loader = build_dataloader(self.cfg, is_train=False, split="test")
        
        # Build evaluator
        self.evaluator = build_evaluator(self.cfg)

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

    def before_train(self, start_time=True):
        self.start_epoch = self.model.resume_or_load_checkpoint(self.cfg, self.optimizer, self.scheduler)
        if start_time:
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
                self.model.save_checkpoint(
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
            self.model.save_checkpoint(
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
                self.model.load_model(f"{self.cfg.MODEL.OUTPUT_DIR}/model")
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

        return list(results.values())[0]
    
    def model_inference(self, input):
        return self.model(input)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

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

class StandardTrainer(AbstractTrainer):
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
        input = batch["img"]
        label = batch["label"]

        if isinstance(input, list):
            input = [x.to(self.device) for x in input]
        else:
            input = input.to(self.device)

        label = label.to(self.device)

        return input, label
        