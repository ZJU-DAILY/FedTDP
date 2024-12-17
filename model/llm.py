from model.model import Model


class LLM(Model):
    def __init__(self, args, llm):
        super().__init__(args=args)
        self.args = args
        self.model = llm

    def forward(self, x):
        return self.adapter(self.model(inputs_embeds=x).last_hidden_state)[0]
