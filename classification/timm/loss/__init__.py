from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, SymmetricCrossEntropy
from .jsd import JsdCrossEntropy
from .asymmetric_loss import AsymmetricLossMultiLabel, AsymmetricLossSingleLabel
from .BiTemperedLogisticLoss import BiTemperedLogisticLoss
from .TaylorCrossEntropyLoss import TaylorCrossEntropyLoss
from .FocalLoss import FocalCosineLoss, FocalLoss
from .ElrLoss import elr_loss