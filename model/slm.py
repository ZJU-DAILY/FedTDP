from model.model import Model


class SLM(Model):
    def __init__(self, args, slm, adapter):
        super().__init__(args=args)
        self.args = args
        self.model = slm
        self.adapter = adapter

    def forward(self, x):
        return self.adapter(self.model(inputs_embeds=x).last_hidden_state)[0]
