import logging
import torch
log = logging.getLogger(__name__)

class Optimizer:
    def __init__(self, model, learning_rate = 0.1, lr_decay = 0.0, weight_decay = 0.0, type = 'Adam'):
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
                weight_decay=weight_decay,
            )

        log.info('Optimizer for %s: %s, learning rate: %s',
                 model._get_name(), type, learning_rate)

    def backprop(self, loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

