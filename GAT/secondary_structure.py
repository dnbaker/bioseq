import torch
import torch.nn as nn

class SecondaryStructurePredictor(nn.Module):
    def __init__(self, in_channels):
        super(SecondaryStructurePredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * in_channels, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

secondary_structure_model = SecondaryStructurePredictor(768)
