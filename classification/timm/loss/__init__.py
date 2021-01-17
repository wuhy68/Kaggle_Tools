from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, SymmetricCrossEntropy
from .jsd import JsdCrossEntropy
from .asymmetric_loss import AsymmetricLossMultiLabel, AsymmetricLossSingleLabel
from .bi_tempered_logistic_loss import BiTemperedLogisticLoss
from .taylor_cross_entropy_loss import TaylorCrossEntropyLoss
from .focal_loss import FocalCosineLoss, FocalLoss

from .loss_factory import create_loss