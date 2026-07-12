"""PyTorch deep learning module."""

from .backprop import ManualBackpropTracker
from .cnn import FinancialCNN
from .losses import FocalLoss, MultiTaskFinancialLoss
from .lstm import FinancialLSTM
from .optimizers import OptimizerComparator, create_optimizer
from .trainer import PyTorchTrainer
from .transformer import FinancialTransformer

__all__ = [
    "FinancialCNN",
    "FinancialLSTM",
    "FinancialTransformer",
    "MultiTaskFinancialLoss",
    "FocalLoss",
    "ManualBackpropTracker",
    "PyTorchTrainer",
    "create_optimizer",
    "OptimizerComparator",
]
