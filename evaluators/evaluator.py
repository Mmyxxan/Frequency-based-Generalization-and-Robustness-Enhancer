import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score

from utils import logger

class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
    
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._y_prob = []

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._y_prob = []

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        # 
        labels_map = ["real", "fake"]
        prob = []
        for batch_prob in mo:
            for idx in torch.topk(batch_prob, k=1).indices.tolist():
                out_prob = torch.softmax(batch_prob, 0)[idx].item()
                if labels_map[idx] == 'real':
                    prob.append(1 - out_prob)
                else:
                    prob.append(out_prob)
        self._y_prob.extend(prob)
        logger.debug(self._y_prob)
        logger.debug(self._y_pred)
        logger.debug(self._y_true)
        # 

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )
        average_precision = 100 * average_precision_score(self._y_true, self._y_prob)

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1
        results["average_precision"] = average_precision

        logger.info(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* average_precision: {average_precision:.2f}%\n"
            f"* accuracy: {acc:.2f}%\n"
            f"* error: {err:.2f}%\n"
            f"* macro_f1: {macro_f1:.2f}%"
        )

        if self.cfg.EVALUATOR.COMPUTE_CONFUSION_MATRIX:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.MODEL.OUTPUT_DIR, "evaluation", "cmat.pt")
            torch.save(cmat, save_path)
            logger.info(f"Confusion matrix is saved to {save_path}")

        return results

class Classification_Output_Probs(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._y_prob = []

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._y_prob = []

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        # 
        labels_map = ["real", "fake"]
        prob = []
        for batch_prob in mo:
            for idx in torch.topk(batch_prob, k=1).indices.tolist():
                out_prob = batch_prob[idx].item()
                if labels_map[idx] == 'real':
                    prob.append(1 - out_prob)
                else:
                    prob.append(out_prob)
        self._y_prob.extend(prob)
        logger.debug(self._y_prob)
        logger.debug(self._y_pred)
        logger.debug(self._y_true)
        # 

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )
        average_precision = 100 * average_precision_score(self._y_true, self._y_prob)

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1
        results["average_precision"] = average_precision

        logger.info(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* average_precision: {average_precision:.2f}%\n"
            f"* accuracy: {acc:.2f}%\n"
            f"* error: {err:.2f}%\n"
            f"* macro_f1: {macro_f1:.2f}%"
        )

        if self.cfg.EVALUATOR.COMPUTE_CONFUSION_MATRIX:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.MODEL.OUTPUT_DIR, "evaluation", "cmat.pt")
            torch.save(cmat, save_path)
            logger.info(f"Confusion matrix is saved to {save_path}")

        return results
