import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


class EMOVO(Dataset):
    """EMOVO: Italian emotional speech database

    Args:
        root_dir (string): Root directory of the dataset.
        transforms (object, optional): Callable class with all transforms to be
            performed. Default: None
        training (boolean, optional): True for training set, False
            for validation set. Default: True
    """
    def __init__(self, root_dir, transforms, training=True):
        self.root_dir = root_dir
        self.transforms = transforms
        self.emotions = [
            'disgusto',   # disgust
            'gioia',      # happiness
            'paura',      # fear
            'rabbia',     # anger
            'sorpresa',   # surprise
            'tristezza',  # sadness
            'neutrale',   # neutral
        ]
        if training:
            self.actors = [
                'm1', 'm2', # male
                'f1', 'f2', # female
            ]
        else:
            self.actors = [
                'm3', 'f3'
            ]
        self.per_actor = 98
        self.sentence_types = [
            'b1', 'b2', 'b3',              # brevi - short
            'l1', 'l2', 'l3', 'l4',        # lunghe - long
            'n1', 'n2', 'n3', 'n4', 'n5',  # nonsense
            'd1', 'd2',                    # domande - questions
        ]

    def __len__(self):
        return self.per_actor * len(self.actors)

    def __getitem__(self, idx):
        actor_idx = idx // self.per_actor
        emotion_idx = idx % self.per_actor // len(self.sentence_types)
        sentence_type_idx = idx % self.per_actor % len(self.sentence_types)

        path = os.path.join(self.root_dir, self.actors[actor_idx],
            '{}-{}-{}.wav'.format(
                self.emotions[emotion_idx][:3],
                self.actors[actor_idx],
                self.sentence_types[sentence_type_idx]
            )
        )
        data, _ = torchaudio.load(path)
        data = torch.mean(data, 0)  # to mono

        if self.transforms is not None:
            data = self.transforms(data)

        data = torch.unsqueeze(data, 0)

        return (data, np.int64(emotion_idx))
