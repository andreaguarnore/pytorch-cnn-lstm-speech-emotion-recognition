from base import BaseDataLoader


class EMOVODataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, validation_split=0.0):
        self.dataset = dataset
        super().__init__(self.dataset, batch_size, shuffle, validation_split)
