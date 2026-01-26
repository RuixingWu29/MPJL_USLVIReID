import torch
import torch.nn as nn
from torch.nn import init


class modal_Classifier(nn.Module):
    def __init__(self, embed_dim, modal_class):
        super(modal_Classifier, self).__init__()
        hidden_size = 1024
        self.first_layer = nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for layer_index in range(7):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True)
            )
            hidden_size = hidden_size // 2  # 512-32-8
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_size, modal_class)

    def forward(self, latent):
        latent = latent.unsqueeze(2)
        hidden = self.first_layer(latent)
        for i in range(7):
            hidden = self.layers[i](hidden)
        style_cls_feature = hidden.squeeze(2)
        modal_cls = self.Liner(style_cls_feature)
        if self.training:
            return modal_cls  # [batch,3]