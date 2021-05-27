import torch
import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB
from torch.utils.data import Dataset
import os


def resample(waveform, sample_rate):
    resampler = Resample(new_freq=sample_rate)
    return resampler(waveform)

def transforms2d(waveform, sample_rate, n_fft, hop_length):
    # stft = Spectrogram(n_fft, hop_length=hop_length)(waveform)
    # mel_spec = MelScale()(stft)
    mel_spec = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )(waveform)
    mel_spec = torch.nan_to_num(mel_spec, 1e-5)

    log_mel_spec = AmplitudeToDB()(mel_spec)

    return log_mel_spec

class EMOVO(Dataset):
    """EMOVO: Italian emotional speech database
    """
    def __init__(self, root_dir, sample_rate=16000,
        audio_length=8, n_fft=2048, hop_length=512, training=True):
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
            training (boolean, optional): True for training set, False
                for validation set. Default: True
        """
        self.root_dir = root_dir
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
        waveform, _ = torchaudio.load(path)
        waveform = torch.mean(waveform, 0)  # to mono
        waveform = resample(waveform, self.sample_rate)
        waveform.resize_(self.sample_rate * self.audio_length)  # cut or pad audio clip

        data = transforms2d(waveform, self.sample_rate, self.n_fft, self.hop_length)
        data = torch.unsqueeze(data, 0)

        return (data, emotion_idx)
