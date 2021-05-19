import torch
from torch.utils.data import Dataset
from scipy.io import wavfile
import os


class EMOVODataset(Dataset):
    """EMOVO: Italian emotional speech database"""

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Root directory of the dataset.
        """
        self.root_dir = root_dir
        self.emotions = [
            'disgusto',   # disgust
            'gioia',      # happiness
            'paura',      # fear
            'rabbia',     # anger
            'sorpresa',   # surprise
            'tristezza',  # sadness
            'neutrale',   # neutral
        ]
        self.actors = [
            'm1', 'm2', 'm3', # male
            'f1', 'f2', 'f3', # female
        ]
        self.per_actor = 98
        self.sentence_types = [
            'b1', 'b2', 'b3', # brevi - short
            'l1', 'l2', 'l3', 'l4', # lunghe - long
            'n1', 'n2', 'n3', 'n4', 'n5', # nonsense
            'd1', 'd2', # domande - questions
        ]

    def __len__(self):
        return self.per_actor * len(self.actors)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        actor_idx = idx // self.per_actor
        emotion_idx = idx % self.per_actor // len(self.sentence_types)
        sentence_type_idx = idx % self.per_actor % len(self.sentence_types)

        path = os.path.join(self.root_dir, self.actors[actor_idx],
            '{}-{}-{}.wav'.format(
                self.emotions[emotion_idx][:3],
                self.actors[actor_idx],
                self.sentence_types[sentence_type_idx])
            )
        print(path)
        _, audiofile = wavfile.read(path)

        sample = {'audiofile': audiofile, 'emotion': emotion_idx}

        return sample
