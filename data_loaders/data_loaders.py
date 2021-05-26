from base import BaseDataLoader
from datasets.datasets import EMOVO


class EMOVODataLoader(BaseDataLoader):
    """
    """
    def __init__(
            self, data_dir, batch_size, raw_data=True, sample_rate=16000, audio_length=8,
            n_fft=2048, hop_length=512, shuffle=True, validation_split=0.0, training=True):
        self.data_dir = data_dir
        self.dataset = EMOVO(self.data_dir, raw_data, sample_rate, audio_length, n_fft, hop_length)
        super().__init__(self.dataset, batch_size, shuffle, validation_split)
