import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class SpeechEmotionModel(BaseModel):
    def __init__(self, emotions):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(128)

        self.flatten1 = nn.Flatten(2)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.flatten2 = nn.Flatten(1)
        self.fc = nn.Linear(256, emotions)

    def forward(self, x):
        x = F.max_pool2d(F.elu(self.bn1(self.conv1(x))), kernel_size=2, stride=2)
        x = F.max_pool2d(F.elu(self.bn2(self.conv2(x))), kernel_size=4, stride=4)
        x = F.max_pool2d(F.elu(self.bn3(self.conv3(x))), kernel_size=4, stride=4)
        x = F.max_pool2d(F.elu(self.bn4(self.conv4(x))), kernel_size=4, stride=4)

        x = self.flatten1(x)
        x = torch.swapaxes(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.flatten2(x)
        x = self.fc(x)

        return F.softmax(x, dim=1)

