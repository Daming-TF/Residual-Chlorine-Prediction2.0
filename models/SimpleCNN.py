import torch
import torch.nn as nn


class Model(nn.Module):
    """
    A Simple Convolution Nerual NetWorker
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.kernel_size = configs.kernel_size
        # self.layer_num = self._upate_layer_num(configs)

        self.module_1 = nn.Sequential(
            nn.Conv1d(1, 8, 3, 1, padding=1), nn.ReLU(),
            nn.Conv1d(8, 16, 3, 1, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 3, 1, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, 1, padding=1), nn.ReLU(),
            nn.Flatten()
        )

        self.predict = nn.Sequential(
            nn.Linear(64*self.seq_len, 1024), nn.ReLU(),
            nn.Linear(1024, self.pred_len)
        )

    def forward(self, input):
        # input: [Batch, Input length, Channel]
        x = input.permute(0, 2, 1)
        x = self.module_1(x)
        x = self.predict(x)
        return x.unsqueeze(2)

    def _upate_layer_num(self, configs):
        if configs.auto_capture:
            return int((configs.seq_len - 1)/(configs.kernel_size-1))+1      # 感受野和kernel_size关系：receptive field=(kernel_size -1)*n+1
        else:
            return configs.layer_num
