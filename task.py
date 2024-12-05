import pandas as pd
from torch.utils.data import DataLoader
from dataset import DataFrameDataset


class Task(object):
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
        if self.args.task == 'anomaly detection':
            self.prompt = 'Task: it is a anomaly detection task. It aims to detect trajectories that deviate significantly from typical movement behaviors. These anomalies could result from unusual user behavior, errors in data collection, or potential malicious activities.'
        elif self.args.task == 'trajectory imputation':
            self.prompt = 'Task: it is a trajectory imputation task. It aims to reconstruct a complete trajectory by predicting or estimating the missing points based on available spatio-temporal data. This often occurs when GPS signals are lost or data collection is interrupted.'
        elif self.args.task == 'noise filtering':
            self.prompt = 'Task: it is a trajectory imputation task. It aims to reconstruct a complete trajectory by predicting or estimating the missing points based on available spatio-temporal data. This often occurs when GPS signals are lost or data collection is interrupted.'
        elif self.args.task == 'stay point detection':
            self.prompt = 'Task: it is a stay point detection task. It aims to identify locations where a moving object (e.g., a person or vehicle) remains within an area for a certain period of time. A stay point typically represents a place of interest, such as a rest stop, home, or office.'
        elif self.args.task == 'map matching':
            self.prompt = 'Task: it is a map matching task. It aims to map the spatio-temporal points to the most likely segments in the road network. This often occurs when the GPS location collected is deviated.'
        elif self.args.task == 'trajectory user link':
            self.prompt = 'Task: it is a trajectory user link task. It aims to link an anonymous trajectory with its corresponding user. These anonymous trajectories are typically used for privacy protection.'
        elif self.args.task == 'travel mode identification':
            self.prompt = 'Task: it is a travel mode identification task. It aims to identify the travel mode of a trajectory based on the moving pattern of data, which is walking, biking, taking the bus, or driving a car.'
        elif self.args.task == 'trajectory simplification':
            self.prompt = 'Task: it is a trajectory simplification task. It aims to reduce the number of spatio-temporal points in a trajectory while preserving its essential shape and features.'
        elif self.args.task == 'trajectory segmentation':
            self.prompt = 'Task: it is a trajectory segmentation task. It aims to divide a trajectory into meaningful segments based on specific criteria such as stay points or travel modes.'
        elif self.args.task == 'trajectory recovery':
            self.prompt = 'Task: it is a trajectory recovery task. It aims to reconstruct a complete trajectory from partially observed or incomplete spatiotemporal points. This often occurs when some parts of the trajectory are missing or unobserved.'

        self.prompt += 'Data: the trajectory data consisting of spatio-temporal points is {}.\n' \
                       'Information: the weather is {} with an average temperature of {}. The road network of the map is {}.\n' \
                       'Format: the output should be the trajectory data.'

    def load_dataset(self, dataset):
        train_dataset, test_dataset, val_dataset, train_ft_dataset, test_ft_dataset, val_ft_dataset = \
            self.split_dataset(dataset=dataset)
        return DataLoader(dataset=DataFrameDataset(features=train_ft_dataset, labels=train_dataset),
                          batch_size=self.args.batch_size), \
            DataLoader(dataset=DataFrameDataset(features=test_ft_dataset, labels=test_dataset),
                       batch_size=self.args.batch_size), \
            DataLoader(dataset=DataFrameDataset(features=val_ft_dataset, labels=val_dataset),
                       batch_size=self.args.batch_size)

    def split_dataset(self, dataset):
        ids = dataset['uid'].unique()
        train_size = int(len(ids) * self.args.train_split)
        test_size = int(len(ids) * self.args.test_split)
        val_size = len(ids) - train_size - test_size
        train_ids = ids[:train_size]
        test_ids = ids[train_size:(train_size + test_size)]
        val_ids = ids[test_size:(train_size + test_size + val_size)]
        return (dataset[dataset['uid'].isin(train_ids)],
                dataset[dataset['uid'].isin(test_ids)],
                dataset[dataset['uid'].isin(val_ids)],
                self.dataset[self.dataset['uid'].isin(train_ids)],
                self.dataset[self.dataset['uid'].isin(test_ids)],
                self.dataset[self.dataset['uid'].isin(val_ids)])
