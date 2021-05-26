import torch
import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB
from torch.utils.data import Dataset
import os


def resample(waveform, sample_rate):
    resampler = Resample(new_freq=sample_rate)
    return resampler(waveform)

def transforms1d(waveform, sample_rate, audio_length):
    return waveform.resize_(sample_rate * audio_length)

def transforms2d(waveform, sample_rate, n_fft, hop_length):
    melspec_transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel_spec = melspec_transform(waveform)

    power_to_db_transform = AmplitudeToDB()
    log_mel_spec = power_to_db_transform(mel_spec)

    return log_mel_spec

class EMOVO(Dataset):
    """EMOVO: Italian emotional speech database
    """
    def __init__(self, root_dir, raw_data=True, sample_rate=16000, audio_length=8, n_fft=2048, hop_length=512):
        """
        Args:
            root_dir (string): Root directory of the dataset.
            raw_data (bool, optional): If true returns raw audio file,
                otherwise the log-mel spectrogram will be extracted with
                the parameters indicated in n_fft and hop_length.
                Default: True
            sample_rate (int, optional): Sample rate to which the signal
                will be resampled. Default: 16000
            audio_length (int, optional): Length in seconds of the audio
                clip, if the clip is longer it will be cut to this
                value, if it is shorter it will be padded with zeros.
                Default: 8
            n_fft (int, optional): Size of FFT for the spectrogram.
                Default: 2048
            hop_length (int, optional): Length of hop between STFT
                windows. Default: 512
        """
        self.root_dir = root_dir
        self.raw_data = raw_data
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.n_fft = n_fft
        self.hop_length = hop_length
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
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

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
        waveform = resample(waveform, self.sample_rate)

        if self.raw_data:
            data = transforms1d(waveform, self.sample_rate, self.audio_length)
        else:
            data = transforms2d(waveform, self.sample_rate, self.n_fft, self.hop_length)

        return (data, emotion_idx)
