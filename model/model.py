import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM
from base import BaseModel


class SpeechEmotionModel(BaseModel):
    def __init__(self, emotions):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=4, stride=4),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=4, stride=4),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        self.flatten = nn.Flatten(2)
        self.lstm = nn.LSTM(128, 32, batch_first=True)
        self.fc = nn.Linear(32, emotions)

    def forward(self, x):
        x = self.convolutions(x)

        x = self.flatten(x)
        x = torch.swapaxes(x, 1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

