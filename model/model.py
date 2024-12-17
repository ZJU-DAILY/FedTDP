from torch import nn


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, x):
        pass
