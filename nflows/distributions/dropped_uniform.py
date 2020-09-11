import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDeepNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers=2):
        super(Net, self).__init__()
        self.layers = []

        if hidden_layers == 0:
            self.layers.append(nn.Linear(in_features, out_features))

        else:
            self.layers.append(nn.Linear(in_features, hidden_features))
            for i in range(hidden_layers-1):
                self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.layers.append(nn.Linear(hidden_features, out_features))

    def forward(self, x):
        for transform in transforms:
            x = F.relu(transform(x))
        return x        


class DroppedUniform(Distribution):
    def __init__(self):

    def _log_prob(self, inputs, context):

    def _sample(self, num_samples, context):