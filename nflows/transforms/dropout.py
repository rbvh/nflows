import torch
import torch.nn as nn
from torch.nn import functional as F

from nflows.transforms.base import Transform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.splines import rational_quadratic
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.base import CompositeTransform, InverseTransform
from nflows.flows.base import Flow

class ProbabilityNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=10, hidden_layers=2):
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
        
class StochasticDropout(Transform):
    '''
    Transform that stochastically sets a subset of features to zero.
    drop_indices is a list of indices of the shape of the features.
    Indices with a 0 are never dropped, all others are dropped with a learned prob.

    The likelihood contribution of the forward transform is uniform
    The likelihood contribution of the inverse transform is given by the probability net

    We don't currently do contexts here.
    '''

    def __init__(self, drop_indices, 
            prob_net_hidden_layers=5,
            prob_net_hidden_features=50,
            rqs_flow_layers=3,
            rqs_num_bins=5, 
            rqs_tails=None, 
            rqs_tail_bound=1.0, 
            rqs_num_blocks=2, 
            rqs_use_residual_blocks=True,
            rqs_random_mask=False,
            rqs_activation=F.relu,
            rqs_dropout_probability=0.0,
            rqs_use_batch_norm=False,
            rqs_min_bin_width = rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
            rqs_min_bin_height = rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
            rqs_min_derivative = rational_quadratic.DEFAULT_MIN_BIN_WIDTH,):
        super(StochasticDropout, self).__init__()

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

        # Arbitrarily set the hidden features to 10x n_probs
        self._prob_net = ProbabilityNet(self._shape, 
            self._n_probs, 
            hidden_features=prob_net_hidden_features, 
            hidden_layers=prob_net_hidden_layers
        )


        # Set up a flow 
        base_dist = BoxUniform(torch.zeros(self._masked_shape), torch.ones(self._masked_shape))
        transforms = []
        for _ in range(rqs_flow_layers):
            transforms.append(RandomPermutation(features=self._masked_shape))
            # Use an inverse transform to ensure that the sampling direction is fast
            transforms.append(InverseTransform(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=self._masked_shape, # The features of the subflow are the n_discrete_dims 
                context_features=self._shape, # The context is the full dim_data input
                hidden_features=rqs_hidden_features,
                num_bins = rqs_num_bins, 
                tails=rqs_tails, 
                tail_bound=rqs_tail_bound, 
                num_blocks=rqs_num_blocks, 
                use_residual_blocks=rqs_use_residual_blocks,
                random_mask=rqs_random_mask,
                activation=rqs_activation,
                dropout_probability=rqs_dropout_probability,
                use_batch_norm=rqs_use_batch_norm,
                min_bin_width=rqs_min_bin_width,
                min_bin_height=rqs_min_bin_height,
                min_derivative=rqs_min_derivative,
            )))
        transform = CompositeTransform(transforms)
        self._flow = Flow(transform, base_dist)


    def forward(self, inputs, context=None):
        # Supplement all zeroes with noise
        inputs_sampled = inputs.clone() # shape = (n_batch, dim_data)
        n_zero_elements = torch.sum(inputs_sampled == 0).item() # shape = n_batch*n_zero_features
        inputs_sampled[inputs_sampled == 0] = torch.rand(n_zero_elements) # shape = (n_batch, dim_data)

        # We now need to compute the likelihood of the inverse transform

        # Get dropout likelihoods from net
        probs = self._prob_net(inputs_sampled) # shape = (n_batch, n_probs)

        # Get a tensor that has a 1 in places where input = 0
        # Then multiply with a descending list of integers
        arranged_zeros = (inputs == 0) * torch.arange(inputs.shape[-1], 0, -1) # shape = (n_batch, n_probs)

        # argmax now always returns the first nonzero element of arranged_zeros
        # i.e. the first zero in inputs
        # Then we get the associated probability index
        # Finally, we subtract 1 to make the indices match up with those of _sample
        prob_index = torch.squeeze(self._drop_indices[torch.argmax(arranged_zeros, -1, keepdim=True)] - 1, -1) # shape = (n_batch)

        # The above code returns an incorrect index if there were no zeroes
        # Set those indices to _n_probs - 1
        prob_index[torch.all(inputs != 0, axis=1)] = self._n_probs - 1 # shape = (n_batch)
        
        # Finally select the probs
        probs_selected = probs[torch.arange(inputs.shape[0]), prob_index]

        return inputs_sampled, torch.log(probs_selected) 

    def inverse(self, inputs, context=None):
        # Figure out the probabilities to drop features
        probs = self._prob_net(inputs)

        # Sample a cutoff
        prob_index = torch.multinomial(probs, 1, replacement=True)
        
        # Zero out the features above the cutoff
        drop_mask = self._drop_indices > prob_index 
        print(self._drop_indices.shape)
        print(prob_index.shape)
        print(drop_mask)
        inputs_dropped = inputs.clone()
        inputs_dropped[drop_mask] = 0

        # prob_index corresponds with the kept dimensions
        # So a 0 means you only keep the zeroes, and a 1 means you keep zeroes and ones
        # Get the values of the chosen likelihoods
        probs_selected = probs[torch.arange(inputs.shape[0]), torch.squeeze(prob_index, -1)]

        return inputs_dropped, torch.log(probs_selected)