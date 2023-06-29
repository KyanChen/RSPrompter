from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmpl.registry import MODELS
from .utils import weighted_loss


@weighted_loss
def uncertainty_regression_loss(pred, target, choice='sooth_l1', beta: float = 1.0) -> Tensor:
    """Smooth L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    y_pred_mean, y_pred_std = pred[..., 0, :, :], pred[..., 1, :, :]
    if choice == 'l2':
        diff = (y_pred_mean - target) ** 2
        diff_std = (torch.abs(y_pred_mean - target) - y_pred_std) ** 2
    elif choice == 'smooth_l1':
        diff = torch.abs(y_pred_mean - target)
        diff = torch.where(diff < beta, 0.5 * diff * diff / beta,
                           diff - 0.5 * beta)
        diff_std = torch.abs(torch.abs(y_pred_mean - target) - y_pred_std)
        diff_std = torch.where(diff_std < beta, 0.5 * diff_std * diff_std / beta,
                               diff_std - 0.5 * beta)
    elif choice == 'l1':
        diff = torch.abs(y_pred_mean - target)
        diff_std = torch.abs(torch.abs(y_pred_mean - target) - y_pred_std)
    else:
        raise NotImplementedError
    loss = diff + diff_std

    return loss


@MODELS.register_module()
class UncertaintyRegressionLoss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 choice='smooth_l1',
                 beta: float = 1.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.choice = choice
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * uncertainty_regression_loss(
            pred,
            target,
            weight,
            choice=self.choice,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
