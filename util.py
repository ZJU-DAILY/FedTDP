import torch.nn.functional as F


class EarlyStopping(object):
    def __init__(self):
        self.patience = 10
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=1), soft_targets, reduction='batchmean') * (
            temperature ** 2)
    return soft_loss
