import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, SymmetricCrossEntropy
from .jsd import JsdCrossEntropy
from .asymmetric_loss import AsymmetricLossMultiLabel, AsymmetricLossSingleLabel
from .bi_tempered_logistic_loss import BiTemperedLogisticLoss
from .taylor_cross_entropy_loss import TaylorCrossEntropyLoss
from .focal_loss import FocalCosineLoss, FocalLoss


def create_loss(args):
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    
    # setup loss function
    train_loss_fn = None
    
    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
        print('JsdCrossEntropyLoss')
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy()
        print('SoftTargetCrossEntropyLoss')
    elif args.loss == 'ce_loss':
        if args.smoothing:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
            print('LabelSmoothingCrossEntropyLoss')
        else:
            train_loss_fn = nn.CrossEntropyLoss()
            print('CrossEntropyLoss')
    elif args.loss == 'tce_loss':
        train_loss_fn = TaylorCrossEntropyLoss()
        print('TaylorCrossEntropyLoss')
    elif args.loss == 'btl_loss':
        train_loss_fn = BiTemperedLogisticLoss(t1=args.t1, t2=args.t2, smoothing=args.smoothing)
        print('BiTemperedLogisticLoss')
    elif args.loss == 'fc_loss':
        train_loss_fn = FocalCosineLoss()
        print('FocalCosineLoss')
    elif args.loss == 'f_loss':
        train_loss_fn = FocalLoss()
        print('FocalLoss')
        
    return train_loss_fn
        