import random
import torch
from peft import LoraConfig, get_peft_model
from torch import nn, optim
from torch.nn import functional
from transformers import AutoModel, AutoTokenizer
from model.slm import SLM
from model.autoencoder import AutoEncoder
from party.party import Party
from util import distillation_loss
import secretflow as sf
from model.llm import LLM
import deepspeed


class Server(Party):
    def __init__(self, args, party):
        super().__init__(args=args, party=party)
        self.clients = None
        self.llm = None
        self.tokenizer = None
        self.model = None
        self.output_layer = None
        self.embedding_layer = None
        self.dropout = None
        self.loss = None
        self.optimizer = None
        self.sigmoid = None
        self.softmax = None

        self.avg_model = None

        self.server_data = None
        self.client_output = None
        self.server_output = None
        self.rp = dict()

        self.load_model()
        self.load_loss()
        self.load_optimizer()

    def secret_sharing(self):
        for i in range(len(self.clients)):
            for j in range(i + 1, len(self.clients)):
                self.clients[i].keys[j] = random.random()
                self.clients[j].keys[i] = random.random()
        ae = AutoEncoder(args=self.args).to(self.args.device)
        k = 0
        for name, param in ae.named_parameters():
            self.rp[name] = k % len(self.clients)
        for i in range(len(self.clients)):
            self.clients[i].rp = sf.PYUObject(device=self.clients[i].party, data=self.rp)

    def load_model(self):
        self.llm = AutoModel.from_pretrained(self.args.llm).to(self.args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.llm)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.llm = get_peft_model(self.llm, LoraConfig(r=8, lora_alpha=32, target_modules=["c_attn", "c_proj"],
                                                       lora_dropout=0.1, bias="none",
                                                       fan_in_fan_out=True)).to(self.args.device)
        if self.args.task == self.args.task == 'trajectory imputation' or self.args.task == 'noise filtering' or self.args.task == 'stay point detection' or self.args.task == 'map matching' or self.args.task == 'trajectory simplification' or self.args.task == 'trajectory segmentation' or self.args.task == 'trajectory recovery':
            self.output_layer = nn.Linear(self.args.max_embed_length * 768, self.args.out * self.args.output_size).to(
                self.args.device)
        elif self.args.task == 'anomaly detection':
            self.output_layer = nn.Linear(self.args.max_embed_length * 768, self.args.out).to(self.args.device)
            self.sigmoid = nn.Sigmoid()
        elif self.args.task == 'traval mode identification' or self.args.task == 'trajectory user link':
            self.output_layer = nn.Linear(self.args.max_embed_length * 768, self.args.out).to(self.args.device)
            self.softmax = nn.Softmax(dim=-1)
        self.embedding_layer = self.llm.get_input_embeddings().to(self.args.device)
        self.dropout = nn.Dropout(p=self.args.drop_out)

        self.avg_slm = get_peft_model(AutoModel.from_pretrained(self.args.slm).to(self.args.device),
                                      LoraConfig(r=8, lora_alpha=32, target_modules=["c_attn", "c_proj"],
                                                 lora_dropout=0.1, bias="none", fan_in_fan_out=True)).to(
            self.args.device)

        self.avg_slm = SLM(args=self.args, slm=self.avg_slm, adapter=self.llm.h[-1])

        self.model = LLM(args=self.args, llm=self.llm)
        self.model, _, _, _ = deepspeed.initialize(args=self.args.deepspeed_config, model=self.model,
                                                   model_parameters=self.model.parameters())

    def load_loss(self):
        if self.args.task == self.args.task == 'trajectory imputation' or self.args.task == 'noise filtering' or self.args.task == 'stay point detection' or self.args.task == 'map matching' or self.args.task == 'trajectory simplification' or self.args.task == 'trajectory segmentation' or self.args.task == 'trajectory recovery':
            self.loss = nn.MSELoss()
        elif self.args.task == 'anomaly detection':
            self.loss = nn.BCELoss()
        elif self.args.task == 'trajectory user link' or self.args.task == 'traval mode identification':
            self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.args.learning_rate)

    def embedding(self, batch_x, a):
        inputs_embeds = []
        for i in range(batch_x.shape[0]):
            prompt = self.clients[0].dataset.task.format(batch_x[i], a[0], a[1], a[2])
            prompt_tokens = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.args.device)
            token_embeds = self.embedding_layer(prompt_tokens)
            token_embeds = functional.pad(token_embeds, (0, 0, 0, self.args.max_embed_length - token_embeds.shape[1]))
            input_embeds = self.dropout(token_embeds)
            inputs_embeds.append(input_embeds)
        return torch.cat(inputs_embeds)

    def train(self):
        for epoch in range(self.args.train_epochs):
            train_loss_slm = 0
            models = list()
            if epoch % self.args.frozen == 0:
                self.server_data = {client.client_num: dict() for client in self.clients}
                self.client_output = {client.client_num: dict() for client in self.clients}
                self.server_output = {client.client_num: dict() for client in self.clients}
            for client in self.clients:
                model, client_train_loss_slm, self.server_data[client.client_num], self.client_output[
                    client.client_num] = client.train_epoch(avg_model=self.avg_slm.state_dict(),
                                                            server_output=sf.PYUObject(device=client.party,
                                                                                       data=self.server_output[
                                                                                           client.client_num]),
                                                            epoch=epoch)
                models.append({name: param for name, param in model.named_parameters()})
                train_loss_slm += client_train_loss_slm
                print('client {} {}'.format(client.client_num, epoch),
                      'train_loss_slm {}'.format(client_train_loss_slm))

            for name, param in self.avg_slm.named_parameters():
                param.data = sum(params[name].data for params in models) / len(models)

            train_loss_slm /= len(self.clients)
            print('server {}'.format(epoch), 'avg_train_loss_slm {}'.format(train_loss_slm))

            for name, client_num in self.rp.items():
                params = list()
                for client in self.clients:
                    params.append(client.party(client.get_params)(n=name))
                param = self.clients[client_num].party(self.clients[client_num].aggregate)(n=name, params=params.to(
                    self.clients[client_num].party))
                for client in self.clients:
                    client.party(client.update)(n=name, p=param.to(client.party))

            self.model.train()
            avg_train_loss_llm = 0
            for client in self.clients:
                train_loss_llm = 0
                for i, (x, y, a) in self.server_data[client.client_num].items():
                    self.optimizer.zero_grad()
                    inputs_embeds = self.embedding(batch_x=x, a=a)
                    output = self.model(inputs_embeds=inputs_embeds).last_hidden_state
                    output = output.view(output.shape[0], -1)
                    if self.args.task == self.args.task == 'trajectory imputation' or self.args.task == 'noise filtering' or self.args.task == 'stay point detection' or self.args.task == 'map matching' or self.args.task == 'trajectory simplification' or self.args.task == 'trajectory segmentation' or self.args.task == 'trajectory recovery':
                        output = self.output_layer(output)
                        output = output.view(output.shape[0], self.args.out, self.args.output_size)
                    elif self.args.task == 'anomaly detection':
                        output = self.sigmoid(self.output_layer(output))
                    elif self.args.task == 'trajectory user link' or self.args.task == 'traval mode identification':
                        output = self.softmax(self.output_layer(output))
                    loss = self.loss(output, y)
                    loss += distillation_loss(output, self.client_output[client.client_num][i])
                    loss.backward()
                    self.optimizer.step()
                    train_loss_llm += loss.item()
                train_loss_llm = round(number=train_loss_llm / len(self.server_data), ndigits=6)
                print('server {}'.format(epoch), 'train_loss_llm {}'.format(train_loss_llm))

                avg_train_loss_llm += train_loss_llm
            print('server {}'.format(epoch), 'avg_train_loss_llm {}'.format(avg_train_loss_llm / len(self.clients)))

            if epoch % self.args.frozen == 0:
                self.model.eval()
                with torch.no_grad():
                    for client in self.clients:
                        for i, (x, y, a) in self.server_data[client.client_num].items():
                            inputs_embeds = self.embedding(batch_x=x, a=a)
                            output = self.model(inputs_embeds=inputs_embeds).last_hidden_state
                            output = output.view(output.shape[0], -1)
                            if self.args.task == self.args.task == 'trajectory imputation' or self.args.task == 'noise filtering' or self.args.task == 'stay point detection' or self.args.task == 'map matching' or self.args.task == 'trajectory simplification' or self.args.task == 'trajectory segmentation' or self.args.task == 'trajectory recovery':
                                output = self.output_layer(output)
                                output = output.view(output.shape[0], self.args.out, self.args.output_size)
                            elif self.args.task == 'anomaly detection':
                                output = self.sigmoid(self.output_layer(output))
                            elif self.args.task == 'trajectory user link' or self.args.task == 'traval mode identification':
                                output = self.softmax(self.output_layer(output))
                            self.server_output[client.client_num][i] = output
