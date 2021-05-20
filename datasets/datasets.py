import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torch.utils.data import Dataset
import os


def _1d_transforms(waveform, sample_rate, audio_length):
    waveform.resize_(sample_rate * audio_length)
    return waveform

def _2d_transforms(waveform, sample_rate):
    melspec_transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
    )
    mel_spec = melspec_transform(waveform)

    power_to_db_transform = AmplitudeToDB()
    log_mel_spec = power_to_db_transform(mel_spec)

    return log_mel_spec

class EMOVO(Dataset):
    """EMOVO: Italian emotional speech database
    """
    def __init__(self, root_dir, audio_length):
        """
        Args:
            root_dir (string): Root directory of the dataset.
        """
        self.root_dir = root_dir
        self.audio_length = audio_length
        self.sample_rate = 48000
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
            'b1', 'b2', 'b3',              # brevi - short
            'l1', 'l2', 'l3', 'l4',        # lunghe - long
            'n1', 'n2', 'n3', 'n4', 'n5',  # nonsense
            'd1', 'd2',                    # domande - questions
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
                self.sentence_types[sentence_type_idx]
            )
        )
        waveform, _ = torchaudio.load(path)
        waveform = torch.mean(waveform, 0)  # to mono

        sample = {
            'audiofile': _1d_transforms(waveform, self.sample_rate, self.audio_length),
            'logmelspectrogram': _2d_transforms(waveform, self.sample_rate),
            'emotion': emotion_idx
        }

        return sample
