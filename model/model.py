import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class Model1D(BaseModel):
    def __init__(self, emotions):
        super().__init__()
        self.conv1d1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1d2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv1d3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv1d4 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)

        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(256 * 500, emotions)

    def forward(self, x):
        x = F.max_pool1d(F.elu(self.conv1d1(x)), kernel_size=4, stride=4)
        x = F.max_pool1d(F.elu(self.conv1d2(x)), kernel_size=4, stride=4)
        x = F.max_pool1d(F.elu(self.conv1d3(x)), kernel_size=4, stride=4)
        x = F.max_pool1d(F.elu(self.conv1d4(x)), kernel_size=4, stride=4)

        x = torch.swapaxes(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x
