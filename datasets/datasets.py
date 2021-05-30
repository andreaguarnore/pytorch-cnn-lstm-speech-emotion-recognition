import os
import numpy as np
import torch
from torch.utils import data
import torchaudio
from torch.utils.data import Dataset


class EMODB(Dataset):
    """EMODB: German emotional speech database

    Args:
        root_dir (string): Root directory of the wav files of the
            dataset.
        transforms (object, optional): Callable class with all
            transforms to be performed. Default: None
        training (boolean, optional): True for training set, False
            for validation set. Default: True
    """
    def __init__(self, root_dir, transforms=None, training=True):
        self.root_dir = root_dir
        self.transforms = transforms
        self.emotions = {k: i for i, k in enumerate([
            'W',  # Ärger (Wut) - anger
            'L',  # Langeweile  - boredom
            'E',  # Ekel        - disgust
            'A',  # Angst       - anxiety/fear
            'F',  # Freude      - happiness
            'T',  # Trauer      - sadness
            'N',  #             - neutral version
        ])}
        self.filenames = os.listdir(self.root_dir)
        delimiter = int(len(self.filenames) * .8)  # delimiter between training and validation
        if training:
            self.filenames = self.filenames[:delimiter]
        else:
            self.filenames = self.filenames[delimiter:]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # load wave file
        path = os.path.join(self.root_dir, self.filenames[idx])
        data, _ = torchaudio.load(path)

        # to mono
        data = torch.mean(data, 0)

        # apply transforms, if needed
        if self.transforms is not None:
            data = self.transforms(data)

        data = torch.unsqueeze(data, 0)

        # parse filename to get emotion
        emotion = self.emotions[path[-6]]  # _x_.wav -> x is the emotion code

        return (data, emotion)

class EMOVO(Dataset):
    """EMOVO: Italian emotional speech database

    Args:
        root_dir (string): Root directory of the dataset.
        transforms (object, optional): Callable class with all
            transforms to be performed. Default: None
        training (boolean, optional): True for training set, False
            for validation set. Default: True
    """
    def __init__(self, root_dir, transforms=None, training=True):
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

        # load wave file
        path = os.path.join(self.root_dir, self.actors[actor_idx],
            '{}-{}-{}.wav'.format(
                self.emotions[emotion_idx][:3],
                self.actors[actor_idx],
                self.sentence_types[sentence_type_idx]
            )
        )
        data, _ = torchaudio.load(path)

        # to mono
        data = torch.mean(data, 0)

        # apply transforms, if needed
        if self.transforms is not None:
            data = self.transforms(data)

        data = torch.unsqueeze(data, 0)

        return (data, np.int64(emotion_idx))
