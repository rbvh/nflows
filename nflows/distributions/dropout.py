import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from nflows.transforms.base import Transform
from nflows.distributions.uniform import BoxUniform
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


class UniformStochasticDropout(Distribution):
    def __init__(self, drop_indices):
        super(UniformStochasticDropout, self).__init__()

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
        
        self._weights = Parameter(torch.Tensor(self._n_probs))
        self._weights.data = torch.rand(self._n_probs)

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

        probs = F.softmax(self._weights)
        return torch.log(probs[prob_index])

    def _sample(self, num_samples, context):
        probs = F.softmax(self._weights)
        cum_probs = torch.cumsum(probs, dim=0)

        # Sample the likelihoods
        larger_than_cumprob = torch.rand(num_samples,1) < cum_probs
        # Do the arange trick to find first nonzero
        selected_index = torch.argmax(larger_than_cumprob*torch.arange(self._n_probs, 0, -1), axis=1)

        # Now sample the output
        output = torch.rand(num_samples, self._shape)
        # Zero out the features above the cutoff
        drop_mask = self._drop_indices > selected_index[:,None]
        output[drop_mask] = 0

        return output

class VariationalStochasticDropout(Distribution):
    '''
    Distribution that stochastically sets a subset of features to zero.
    drop_indices is a list of indices of the shape of the features.
    Indices with a 0 are never dropped, all others are dropped with a learned prob.
    Every separate index in drop_indices gets its own probability

    We don't currently do contexts here.
    '''

    def __init__(self, drop_indices, 
            prob_net_hidden_layers=5,
            prob_net_hidden_features=50,
            rqs_flow_layers=3,
            rqs_hidden_features=50,
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
        super(VariationalStochasticDropout, self).__init__()

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

        self._prob_net = ProbabilityNet(self._shape, 
            self._n_probs, 
            hidden_features=prob_net_hidden_features, 
            hidden_layers=prob_net_hidden_layers
        )

        # Set up a flow 
        base_dist = BoxUniform(torch.zeros(self._shape), torch.ones(self._shape))
        transforms = []
        for _ in range(rqs_flow_layers):
            transforms.append(RandomPermutation(features=self._shape))
            # Use an inverse transform to ensure that the sampling direction is fast
            transforms.append(InverseTransform(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=self._shape,
                context_features=self._shape, # features and context_features are both the feature size
                hidden_features=rqs_hidden_features,
                num_bins=rqs_num_bins, 
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

        #print("The index of the selected probability")
        #print("This is also the highest label in drop_indices that is kept")
        #print(prob_index)

        # Now compute the dropout probabilities - first sample the inverse from the conditional flow
        # Note - we have moved the batch size of the sample into the context. 
        # i.e rather than generating a sample for every context, we instead generate a single sample for a batch of contexts
        noise, log_probs_flow = self._flow.sample_and_log_prob(1, context=inputs) # shape = (n_batch, 1, n_discrete_dims), (n_batch, 1)
        # Squeeze out the unit dim from the noise & flowprob
        noise = torch.squeeze(noise, dim=1)
        log_probs_flow = torch.squeeze(log_probs_flow, dim=1)
        
        # Get dropout likelihoods from net
        dropout_probs = self._prob_net(noise)

        # Pick out the selected probs
        dropout_probs_selected = dropout_probs[torch.arange(inputs.shape[0]), prob_index]

        return torch.log(dropout_probs_selected) - log_probs_flow

    def _sample(self, num_samples, context):
        noise = torch.rand(num_samples, self._shape)

        # Figure out the probabilities to drop features
        probs = self._prob_net(noise)

        # Select an index
        cum_probs = torch.cumsum(probs, dim=1)
        larger_than_cum_probs = torch.rand(num_samples,1) < cum_probs

        # Do the arange trick to find first nonzero
        selected_index = torch.argmax(larger_than_cum_probs*torch.arange(self._n_probs, 0, -1), axis=1)
        
        # Get the values of the selected llhs
        selected_probs = probs[torch.arange(num_samples), selected_index]

        # Find the index of the first true
        drop_mask = self._drop_indices > selected_index[:,None]
        noise[drop_mask] = 0

        return noise