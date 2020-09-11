import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.distributions.base import Distribution

class ProbabilityNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers=3):
        super(ProbabilityNet, self).__init__()
        self._layers = []

        if hidden_layers == 0:
            self._layers.append(nn.Linear(in_features, out_features))

        else:
            self._layers.append(nn.Linear(in_features, hidden_features))
            for i in range(hidden_layers-1):
                self._layers.append(nn.Linear(hidden_features, hidden_features))
            self._layers.append(nn.Linear(hidden_features, out_features))

    def forward(self, x):
        for layer in self._layers:
            x = F.relu(layer(x))
        return F.softmax(x)        


class DroppedUniform(Distribution):
    '''
    Uniform distribution that stochastically sets a subset of features to zero.
    drop_indices is a list of indices of the shape of the features.
    Indices with a 0 are never dropped, all others are dropped with a learned prob.

    We don't currently do contexts here.
    '''
    def __init__(self, drop_indices, hidden_layers=3):
        super().__init__()

        self._shape = drop_indices.shape[0]
        self._n_probs = torch.unique(drop_indices).shape[0]
        self._drop_indices = drop_indices

        # Add one for the option to not drop anything in case there is no index 0
        if (torch.all(drop_indices != 0)):
            self._n_probs += 1
        
        if self._n_probs <= 0:
            raise ValueError(
                "No droppable features included."
            )

        if (self._n_probs != torch.max(drop_indices).item()+1):
            raise ValueError(
                "Make sure all indices between 0 and the maximum are included."
            )

        # Arbitrarily set the hidden features to 2x n_probs
        self._prob_net = ProbabilityNet(self._shape, self._n_probs, 2*self._n_probs, hidden_layers=hidden_layers)

    def _log_prob(self, inputs, context):
        # Compute the log_prob with a MC sample of the inverse
        # Supplement all zeroes with noise
        inputs_supplemented = inputs.clone()
        n_zero_elements = torch.sum(inputs_supplemented == 0).item()
        inputs_supplemented[inputs_supplemented == 0] = torch.rand(n_zero_elements)

        # Compute likelihoods
        probs = self._prob_net(inputs_supplemented)

        # 
        test = torch.searchsorted(inputs, torch.zeros(inputs.shape[0]))
        print(test)


    def _sample(self, num_samples, context):
        # First sample unit box noise
        noise = torch.rand(num_samples, self._shape)

        # Figure out the probabilities to drop features
        drop_probs = self._prob_net(noise)

        # Sample a cutoff
        drop_indices_cutoff = torch.multinomial(drop_probs, 1, replacement=True)
        
        # Zero out the features above the cutoff
        drop_mask = self._drop_indices > drop_indices_cutoff 
        noise[drop_mask] = 0

        return noise
