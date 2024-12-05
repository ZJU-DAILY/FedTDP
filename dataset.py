from torch.utils.data import Dataset
import torch
from task import Task
import pandas as pd


class DataSet(object):
    def __init__(self, args, num):
        self.args = args
        self.dataset = pd.read_csv(num + self.args.data)
        time_max = self.dataset['time'].max()
        time_min = self.dataset['time'].min()
        lon_max = self.dataset['lon'].max()
        lon_min = self.dataset['lon'].min()
        lat_max = self.dataset['lat'].max()
        lat_min = self.dataset['lat'].min()
        self.dataset['time'] = (self.dataset['time'] - time_min) / (time_max - time_min)
        self.dataset['lon'] = (self.dataset['lon'] - lon_min) / (lon_max - lon_min)
        self.dataset['lat'] = (self.dataset['lat'] - lat_min) / (lat_max - lat_min)
        self.task = Task(args=args, num=num)

    def load_dataset(self):
        return self.task.load_dataset(dataset=self.dataset)


class DataFrameDataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = features
        self.labels = labels
        self.ids = features['uid'].unique()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return torch.tensor(
            self.features[self.features['uid'] == self.ids[index]].drop(columns=['uid', 'weather'], axis=1).values).to(
            torch.float32), torch.tensor(
            self.labels[self.labels['uid'] == self.ids[index]].drop(columns='uid', axis=1).values).to(
            torch.float32), self.features[self.features['uid'] == self.ids[index]]['in'].values
