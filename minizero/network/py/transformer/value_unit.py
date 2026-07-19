import torch.nn as nn
import torch.nn.functional as F


class TValueNetwork(nn.Module):
    def __init__(self, emb_size, input_channel_height):
        super().__init__()
        self.input_channel_height = input_channel_height
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # squeeze
        self.fc = nn.Sequential(
            nn.Linear(emb_size, emb_size * 2),
            nn.SiLU(),
            nn.Linear(emb_size * 2, 1),
        )
        self.tanh = nn.Tanh()

    def rearrange_to_conv(self, x):
        # "b (h w) c -> b c h w"
        b, _, c = x.shape
        return x.view(b, self.input_channel_height, -1, c).permute(0, 3, 1, 2)

    def forward(self, x):
        if x.dim() == 3:
            x = self.rearrange_to_conv(x)
        b, c, _, _ = x.size()
        x = self.avg_pool(x).view(b, c)
        x = self.fc(x)
        x = self.tanh(x)
        return x


class ValueNetwork(nn.Module):
    def __init__(
        self, num_channels, channel_height, channel_width, num_output_channels
    ):
        super(ValueNetwork, self).__init__()
        self.channel_height = channel_height
        self.channel_width = channel_width
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(channel_height * channel_width, num_output_channels)
        self.fc2 = nn.Linear(num_output_channels, 1)
        self.tanh = nn.Tanh()

    def rearrange_to_conv(self, x):
        # "b (h w) c -> b c h w"
        b, _, c = x.shape
        return x.view(b, self.channel_height, self.channel_width, c).permute(0, 3, 1, 2)

    def forward(self, x):
        if x.dim() == 3:
            x = self.rearrange_to_conv(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.contiguous().view(-1, self.channel_height * self.channel_width)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x
