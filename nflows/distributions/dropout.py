import torch
import torch.nn as nn
from torch.nn import functional as F

from nflows.transforms.base import Transform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.splines import rational_quadratic
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.base import CompositeTransform, InverseTransform
from nflows.distributions.base import Distribution
from nflows.flows.base import Flow

class ProbabilityNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers=0):
        super(ProbabilityNet, self).__init__()
        layers = []

        if hidden_layers == 0:
            layers.append(nn.Linear(in_features, out_features))

        else:
            layers.append(nn.Linear(in_features, hidden_features))
            for i in range(hidden_layers-1):
                layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.Linear(hidden_features, out_features))

        self._layers = nn.ModuleList(layers)

    def forward(self, x):
        # Relu on the first n-1 layers, softmax on the last
        for layer in self._layers[:-1]:
            x = F.relu(layer(x))
        return F.softmax(self._layers[-1](x), dim=-1)

class StochasticDropout(Distribution):
    '''
    Distribution that stochastically sets a subset of features to zero.
    drop_indices is a list of indices of the shape of the features.
    Indices with a 0 are never dropped, all others are dropped with a learned prob.
    Every separate index in drop_indices gets its own probability

    We don't currently do contexts here.
    '''

    def __init__(self, drop_indices, hidden_layers=5):
        super(StochasticDropout, self).__init__()

        self._shape = drop_indices.shape[0]
        self._n_probs = torch.unique(drop_indices).shape[0]
        self._drop_indices = drop_indices
        self._n_MC = 200

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

        # Arbitrarily set the hidden features to 10x n_probs
        self._prob_net = ProbabilityNet(self._shape, self._n_probs, 10*self._n_probs, hidden_layers=hidden_layers)

    def _log_prob(self, inputs, context):
        # We first locate the index of the first zero element
        # Do this by:
        # 1) inputs == 0 gives a tensor of booleans
        # 2) multiply with a tensor of decreasing numbers
        # 3) get the argmax
        first_zero_index = torch.argmax((inputs == 0) * torch.arange(inputs.shape[-1], 0, -1), -1, keepdim=True) # shape = (n_batch, 1)
        # Next, compute the highest index kept from _drop_indices 
        # Have to subtract 1 to make the indices match up with those of _sample
        # If there were no zeroes, the index should be self._n_probs - 1
        prob_index = torch.squeeze(self._drop_indices[first_zero_index] - 1, -1) # shape = (n_batch)
        prob_index[torch.all(inputs != 0, axis=1)] = self._n_probs - 1 # shape = (n_batch)

        '''
        print("The index of the selected probability")
        print("This is also the highest label in drop_indices that is kept")
        print(prob_index)
        '''
        
        # Now, we compute the dropout probabilities
        # They are evaluated n_MC times
        # To that end, add a new dim to the input and replicate the data n_MC times
        inputs_filled = inputs.clone()[:,None,:].repeat(1,self._n_MC,1) # shape = (n_batch, n_MC, dim_data)
        # Next, isolate all zero elements in inputs_filled and replace with noise
        n_zero_elements = torch.sum(inputs_filled == 0).item() # shape = n_batch**n_MC*n_zero_features
        inputs_filled[inputs_filled == 0] = torch.rand(n_zero_elements) # shape = (n_batch, n_MC, dim_data)
        # Get dropout likelihoods from net
        probs = self._prob_net(inputs_filled) # shape = (n_batch, n_probs)
        # Take a mean over the n_MC dimensions
        mean_probs = probs.mean(dim=1)
        
        # Finally select the probs
        mean_probs_selected = mean_probs[torch.arange(inputs.shape[0]), prob_index]

        return torch.log(mean_probs_selected) 

    def _sample(self, num_samples, context):   
        return 0