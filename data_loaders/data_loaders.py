from torchvision import datasets, transforms
from base import BaseDataLoader
from datasets import *


class EMOVODataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = datasets.EMOVO(self.data_dir, training=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
