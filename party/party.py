import torch
import secretflow as sf

torch.manual_seed(42)


class Party(object):
    def __init__(self, args, party: sf.SPU):
        self.args = args
        self.party = party

    def load_model(self):
        pass
