import torch
from torch import nn
from model.model import Model


class GELU(Model):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class MLP(Model):
    def __init__(self, args):
        super(MLP, self).__init__(args=args)
        self.fc1 = nn.Linear(self.args.input_size, self.args.hidden_size)
        self.fc2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.fc3 = nn.Linear(self.args.hidden_size, self.args.output_size)
        self.gelu = GELU(args=args)

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.fc3(x)
        return x


class AutoEncoder(Model):
    def __init__(self, args):
        super(AutoEncoder, self).__init__(args=args)
        self.encoder = MLP(args=args)
        self.decoder = MLP(args=args)

    def forward(self, x):
        return self.decoder(self.encoder(x))
