import json
import os
from .BaseDataset import BaseDataset


class open_ended_dataset(BaseDataset):
    def __init__(self, dataset_name='open_ended_dataset', type='religious_ideology', data_root='./data/', **kwargs):
        """
            type: 'gender', 'race', 'religious_ideology', 'profession-emo', 'profession-nohuman'
        """
        super().__init__(dataset_name, **kwargs)
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.type = type
        if "profession" in self.type:
            self.input_json = os.path.join(self.data_root,
                                           'open_ended/profession/annotation_{}.json'.format(self.type.split('-')[1]))
        else:
            self.input_json = os.path.join(self.data_root, 'open_ended/{}/annotation.json'.format(self.type))
        with open(self.input_json, 'r') as file:
            self.data = json.load(file)
        self.num_samples = len(self.data)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.num_samples:
            sample = self.data[self.index]
            sample['images'][0] = os.path.join(self.data_root, sample['images'][0])
            self.index += 1
            return sample
        else:
            raise StopIteration
