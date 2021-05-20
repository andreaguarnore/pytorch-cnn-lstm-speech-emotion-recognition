from base import BaseDataLoader
from datasets.datasets import EMOVO


class EMOVODataLoader(BaseDataLoader):
    """
    """
    def __init__(
            self, data_dir, audio_length, batch_size, shuffle=True,
            validation_split=0.0, training=True):
        self.data_dir = data_dir
        self.dataset = EMOVO(self.data_dir, audio_length)
        super().__init__(self.dataset, batch_size, shuffle, validation_split)
