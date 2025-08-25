"""
Lightweight wrapper around PyTorch optimizers.

Automatically selects optimizer type and applies common settings.
"""

import logging
import torch

log = logging.getLogger(__name__)


class Optimizer:
    """
    Wrapper for torch.optim.* that:
        1. Filters out non-trainable parameters.
        2. Instantiates the requested optimizer.
        3. Exposes a single back-prop step method.
    """

    def __init__(self,
                 model,
                 learning_rate: float = 0.1,
                 lr_decay: float = 0.0,
                 weight_decay: float = 0.0,
                 type: str = 'Adam'):
        """
        Parameters
        ----------
        model : nn.Module
            The model whose parameters will be optimized.
        learning_rate : float
            Initial learning rate.
        lr_decay : float
            Learning-rate decay (used only by Adagrad).
        weight_decay : float
            L2 weight-decay coefficient.
        type : {'Adagrad', 'Adam', 'SGD'}
            Optimizer algorithm to use.
        """

        # Collect only parameters that require gradients
        self.params = list(filter(lambda p: p.requires_grad, model.parameters()))

        if type == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(
                self.params,
                lr=learning_rate,
                lr_decay=lr_decay,
                weight_decay=weight_decay
            )
        elif type == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.params,
                lr=learning_rate,
                weight_decay=weight_decay,
                amsgrad=True
            )
        elif type == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.params,
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {type}")

        log.info('Optimizer for %s: %s, learning rate: %s',
                 model._get_name(), type, learning_rate)

    def backprop(self, loss: torch.Tensor) -> None:
        """
        Perform one back-propagation step.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss tensor.
        """
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
