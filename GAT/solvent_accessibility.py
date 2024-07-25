import torch
import torch.nn as nn

class SolventAccessibilityPredictor(nn.Module):
    def __init__(self, in_channels):
        super(SolventAccessibilityPredictor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64, 512)
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

solvent_accessibility_model = SolventAccessibilityPredictor(768)
