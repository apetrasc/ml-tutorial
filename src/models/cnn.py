import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
rng = np.random.RandomState(1234)
random_state = 42
class Conv(nn.Module):
    def __init__(self, filter_shape, function=lambda x: x, stride=(1, 1), padding=0):
        super().__init__()
        # Heの初期化
        # filter_shape: (出力チャンネル数)x(入力チャンネル数)x(縦の次元数)x(横の次元数)
        fan_in = filter_shape[1] * filter_shape[2] * filter_shape[3]
        fan_out = filter_shape[0] * filter_shape[2] * filter_shape[3]

        self.W = nn.Parameter(torch.tensor(rng.normal(
                        0,
                        np.sqrt(2/fan_in),
                        size=filter_shape
                    ).astype('float32')))

        # バイアスはフィルタごとなので, 出力フィルタ数と同じ次元数
        self.b = nn.Parameter(torch.tensor(np.zeros((filter_shape[0]), dtype='float32')))

        self.function = function  # 活性化関数
        self.stride = stride  # ストライド幅
        self.padding = padding  # パディング

    def forward(self, x):
        u = F.conv2d(x, self.W, bias=self.b, stride=self.stride, padding=self.padding)
        return self.function(u)


conv_net = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),   # 32x32x3 -> 32x32x64
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),                  # 32x32x64 -> 16x16x64

    nn.Conv2d(64, 128, 3, padding=1), # 16x16x64 -> 16x16x128
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2),                  # 16x16x128 -> 8x8x128

    nn.Conv2d(128, 256, 3, padding=1),# 8x8x128 -> 8x8x256
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.MaxPool2d(2),                  # 8x8x256 -> 4x4x256

    nn.Conv2d(256, 512, 3, padding=1),# 4x4x256 -> 4x4x512
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.MaxPool2d(2),                  # 4x4x512 -> 2x2x512

    nn.Flatten(),
    nn.Linear(2*2*512, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)