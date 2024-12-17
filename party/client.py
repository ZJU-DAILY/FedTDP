import math
import random
import torch
from torch import nn
from torch.nn import functional
from torch import optim
from party.party import Party
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from dataset import DataSet
from util import distillation_loss
import copy
from model.slm import SLM
from model.autoencoder import AutoEncoder
import secretflow as sf


class Client(Party):
    def __init__(self, args, client_num, server, party):
        super().__init__(args=args, party=party)
        self.server = server
        self.clients = None
        self.client_num = client_num
        self.dataset = None
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.laplace = None
        self.loss = None
        self.optimizer_ae = None
        self.optimizer_slm = None

        self.ae = None
        self.slm = None
        self.tokenizer = None
        self.prompt = None
        self.model = None
        self.embedding_layer = None
        self.output_layer = None
        self.soft_prompt_embeddings = None
        self.dropout = None
        self.sigmoid = None
        self.softmax = None

        self.server_data = None
        self.client_output = None
        self.last_model = dict()
        self.nm = None
        self.keys = dict()
        self.rp = dict()

        self.load_dataset()
        self.load_model()
        self.load_loss()
        self.load_optimizer()

    def load_dataset(self):
        self.dataset = DataSet(args=self.args, num=self.client_num)
        self.train_loader, self.test_loader, self.val_loader = self.dataset.load_dataset()

    def load_model(self):
        self.ae = AutoEncoder(args=self.args).to(self.args.device)
        self.slm = AutoModel.from_pretrained(self.args.llm).to(self.args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.llm)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.slm = get_peft_model(
            self.slm, LoraConfig(r=8, lora_alpha=32, target_modules=["c_attn", "c_proj"],
                                 lora_dropout=0.1, bias="none", fan_in_fan_out=True)).to(self.args.device)
        self.embedding_layer = self.slm.get_input_embeddings().to(self.args.device)
        if self.args.task == self.args.task == 'trajectory imputation' or self.args.task == 'noise filtering' or self.args.task == 'stay point detection' or self.args.task == 'map matching' or self.args.task == 'trajectory simplification' or self.args.task == 'trajectory segmentation' or self.args.task == 'trajectory recovery':
            self.output_layer = nn.Linear(self.args.max_embed_length * 768, self.args.out * 3).to(self.args.device)
        elif self.args.task == 'anomaly detection':
            self.output_layer = nn.Linear(self.args.max_embed_length * 768, self.args.out).to(self.args.device)
            self.sigmoid = nn.Sigmoid()
        elif self.args.task == 'traval mode identification' or self.args.task == 'trajectory user link':
            self.output_layer = nn.Linear(self.args.max_embed_length * 768, self.args.out).to(self.args.device)
            self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=self.args.drop_out)
        self.nm = self.party(device=self.party, data=round(self.args.m * sum(1 for _ in self.slm.parameters())))

        self.model = SLM(args=self.args, slm=self.slm, adapter=self.server.model.h[-1])

    def load_loss(self):
        if self.args.task == self.args.task == 'trajectory imputation' or self.args.task == 'noise filtering' or self.args.task == 'stay point detection' or self.args.task == 'map matching' or self.args.task == 'trajectory simplification' or self.args.task == 'trajectory segmentation' or self.args.task == 'trajectory recovery':
            self.loss = nn.MSELoss()
        elif self.args.task == 'anomaly detection':
            self.loss = nn.BCELoss()
        elif self.args.task == 'trajectory user link' or self.args.task == 'traval mode identification':
            self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        self.optimizer_ae = optim.Adam(params=self.ae.parameters(), lr=self.args.learning_rate)
        self.optimizer_slm = optim.Adam(params=self.model.parameters(), lr=self.args.learning_rate)

    def embedding(self, batch_x, a):
        inputs_embeds = []
        for i in range(batch_x.shape[0]):
            prompt = self.dataset.task.prompt.format(batch_x[i], a[0], a[1], a[2])
            prompt_tokens = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.args.device)
            token_embeds = self.embedding_layer(prompt_tokens)
            token_embeds = functional.pad(token_embeds, (0, 0, 0, self.args.max_embed_length - token_embeds.shape[1]))
            input_embeds = self.dropout(token_embeds)
            inputs_embeds.append(input_embeds)
        return torch.cat(inputs_embeds)

    def train_epoch(self, avg_model, server_output, epoch):
        self.model.load_state_dict(avg_model)
        if epoch > 0:
            crs = dict()
            rs = dict()
            selected = dict()
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                crs[name] = torch.abs(param.data - self.last_model[name].data) / torch.abs(self.last_model[name].data)
                for _ in range(len(crs[name].shape)):
                    crs[name] = sum(crs[name])
                if crs[name] == float('inf') and len(selected) < self.nm:
                    crs[name] = 0
                    selected[name] = 0
                    param.requires_grad = True

            for name, param in self.model.named_parameters():
                rs[name] = crs[name] / sum(crs.values())

            for _ in range(self.nm - len(selected)):
                seed = random.random()
                ran = 0
                for name, param in self.model.named_parameters():
                    rs[name] = rs[name] * math.prod(([p if p != 0 else 1 for p in selected.values()])) / (
                            1 - sum(selected.values()))

                for name, param in self.model.named_parameters():
                    ran += rs[name]
                    if seed < ran:
                        selected[name] = rs[name]
                        param.requires_grad = True
                        break
        for name, param in self.model.named_parameters():
            self.last_model[name] = copy.deepcopy(param)

        self.ae.train()
        self.model.train()
        train_loss_ae = 0
        train_loss_slm = 0
        if epoch % self.args.frozen == 0:
            self.server_data = dict()
            self.client_output = dict()
        for i, (batch_x, batch_y, a) in enumerate(self.train_loader):
            batch_x = batch_x.to(self.args.device)
            batch_y = batch_y.to(self.args.device)

            self.optimizer_ae.zero_grad()
            loss_ae = self.loss(self.ae(batch_x), batch_x)
            loss_ae.backward()
            self.optimizer_ae.step()
            train_loss_ae += loss_ae.item()

            self.optimizer_slm.zero_grad()
            inputs_embeds = self.embedding(batch_x=batch_x, a=a)
            output = self.model(inputs_embeds=inputs_embeds).last_hidden_state
            output = output.view(output.shape[0], -1)
            if self.args.task == self.args.task == 'trajectory imputation' or self.args.task == 'noise filtering' or self.args.task == 'stay point detection' or self.args.task == 'map matching' or self.args.task == 'trajectory simplification' or self.args.task == 'trajectory segmentation' or self.args.task == 'trajectory recovery':
                output = self.output_layer(output)
                output = output.view(output.shape[0], self.args.out, 3)
            elif self.args.task == 'anomaly detection':
                output = self.sigmoid(self.output_layer(output))
            elif self.args.task == 'trajectory user link' or self.args.task == 'traval mode identification':
                output = self.softmax(self.output_layer(output))
            loss_slm = self.loss(output, batch_y)

            if len(server_output) != 0:
                loss_slm += distillation_loss(self.ae.encoder(output), server_output[i])
            loss_slm.backward()
            self.optimizer_slm.step()
            train_loss_slm += loss_slm.item()
        train_loss_ae = round(number=train_loss_ae / len(self.train_loader), ndigits=6)
        train_loss_slm = round(number=train_loss_slm / len(self.train_loader), ndigits=6)
        print('client {} {}'.format(self.client_num, epoch), 'train_loss_ae {}'.format(train_loss_ae),
              'train_loss_slm {}'.format(train_loss_slm))

        if epoch % self.args.frozen == 0:
            self.ae.eval()
            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, a) in enumerate(self.train_loader):
                    batch_x = batch_x.to(self.args.device)
                    batch_y = batch_y.to(self.args.device)
                    inputs_embeds = self.embedding(batch_x=batch_x, a=a)
                    output = self.model(inputs_embeds=inputs_embeds).last_hidden_state
                    output = output.view(output.shape[0], -1)
                    if self.args.task == self.args.task == 'trajectory imputation' or self.args.task == 'noise filtering' or self.args.task == 'stay point detection' or self.args.task == 'map matching' or self.args.task == 'trajectory simplification' or self.args.task == 'trajectory segmentation' or self.args.task == 'trajectory recovery':
                        output = self.output_layer(output)
                        output = output.view(output.shape[0], self.args.out, 3)
                    elif self.args.task == 'anomaly detection':
                        output = self.sigmoid(self.output_layer(output))
                    elif self.args.task == 'trajectory user link' or self.args.task == 'traval mode identification':
                        output = self.softmax(self.output_layer(output))
                    self.server_data[i] = (self.ae.encoder(batch_x), self.ae.encoder(batch_y), w)
                    self.client_output[i] = self.ae.encoder(output)
        return self.model, train_loss_slm, sf.PYUObject(device=self.server.party, data=self.server_data), sf.PYUObject(
            device=self.server.party, data=self.client_output)

    def get_params(self, n):
        for name, param in self.ae.named_parameters():
            if name == n:
                for num, key in self.keys.items():
                    if self.client_num > num:
                        param.data += key
                    else:
                        param.data -= key
                return param

    def aggregate(self, n, params):
        for name, param in self.ae.named_parameters():
            if name == n:
                for p in params:
                    param.data += p.data
                param.data /= len(self.clients)
                return param

    def update(self, n, p):
        for name, param in self.ae.named_parameters():
            if name == n:
                param.data = p.data
