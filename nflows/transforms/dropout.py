import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from nflows.distributions.uniform import BoxUniform
from nflows.transforms.base import Transform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.splines import rational_quadratic
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.base import CompositeTransform, InverseTransform
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

class UniformStochasticDropout(Transform):
    '''
    Transform that stochastically sets a subset of features to zero.
    drop_indices is a list of indices of the shape of the features.
    Indices with a 0 are never dropped, all others are dropped with a regressed prob.

    The likelihood contribution of the forward transform is uniform
    The likelihood contribution of the inverse transform is given by the regressed probs

    Context is currently ignored
    '''
    def __init__(self, drop_indices):
        super(UniformStochasticDropout, self).__init__()

        self.register_buffer("_n_probs", torch.unique(drop_indices).shape[0])
        self.register_buffer("_drop_indices", drop_indices)
        
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
        
        self.register_parameter(_weights, torch.Tensor(self._n_probs))
        self._weights.data = torch.rand(self._n_probs)
    
    def forward(self, inputs, context=None):
        # We first locate the index of the first zero element
        # Do this by:
        # 1) inputs == 0 gives a tensor of booleans
        # 2) multiply with a tensor of decreasing numbers
        # 3) get the argmax
        zero_mask = inputs == 0
        first_zero_index = torch.argmax(zero_mask * torch.arange(inputs.shape[-1], 0, -1), -1, keepdim=True) # shape = (n_batch, 1)

        # Next, compute the highest index kept from _drop_indices 
        # Have to subtract 1 to make the indices match up with those of _sample
        # If there were no zeroes, the index should be self._n_probs - 1
        probs_dropout_index = torch.squeeze(self._drop_indices[first_zero_index] - 1, -1) # shape = (n_batch)
        probs_dropout_index[torch.all(inputs != 0, axis=1)] = self._n_probs - 1 # shape = (n_batch)

        # Compute the probs
        log_probs_dropout_selected = torch.log(F.softmax(self._weights)[probs_dropout_index])
        
        # Clone inputs and append random noise
        outputs = inputs.clone()
        outputs[zero_mask] = torch.rand(inputs[zero_mask].shape)

        return outputs, log_probs_dropout_selected

    def inverse(self, inputs, context=None):
        batch_size = inputs.shape[0]

        # Probabilities to drop features
        probs_dropout = F.softmax(self._weights)
        cum_probs_dropout = torch.cumsum(probs_dropout, dim=0)
    
        # Select a drop probability
        larger_than_cum_probs_dropout = torch.rand(batch_size, 1) < cum_probs_dropout
        # Do the arange trick to find first nonzero
        drop_index_selected = torch.argmax(larger_than_cum_probs_dropout*torch.arange(self._n_probs, 0, -1), axis=1)
        log_probs_dropout_selected = torch.log(probs_dropout[drop_index_selected])

        # Clone the input
        outputs = inputs.clone()

        # Zero out the features above the cutoff
        zero_mask = self._drop_indices > drop_index_selected[:,None]
        outputs[zero_mask] = 0

        return outputs, -log_probs_dropout_selected

class VariationalStochasticDropout(Transform):
    '''
    Transform that stochastically sets a subset of features to zero.
    drop_indices is a list of indices of the shape of the features.
    Indices with a 0 are never dropped, all others are dropped with a conditional probability

    The likelihood contribution of the forward transform is variational
    The likelihood contribution of the inverse transform is given by the conditional probabilities

    Context is currently ignored
    '''

    def __init__(self, drop_indices, 
        prob_net_hidden_layers=5,
        prob_net_hidden_features=50,
        rqs_flow_layers=5,
        rqs_hidden_features=50,
        rqs_num_bins=10, 
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

    def forward(self, inputs, context=None):
        # We first locate the index of the first zero element
        # Do this by:
        # 1) inputs == 0 gives a tensor of booleans
        # 2) multiply with a tensor of decreasing numbers
        # 3) get the argmax
        zero_mask = inputs == 0
        first_zero_index = torch.argmax(zero_mask * torch.arange(inputs.shape[-1], 0, -1), -1, keepdim=True) # shape = (n_batch, 1)

        # Next, compute the highest index kept from _drop_indices 
        # Have to subtract 1 to make the indices match up with those of _sample
        # If there were no zeroes, the index should be self._n_probs - 1
        prob_index = torch.squeeze(self._drop_indices[first_zero_index] - 1, -1) # shape = (n_batch)
        prob_index[torch.all(inputs != 0, axis=1)] = self._n_probs - 1 # shape = (n_batch)

        # Now compute the dropout probabilities - first sample the inverse from the conditional flow
        # Note - we have moved the batch size of the sample into the context. 
        # i.e rather than generating a sample for every context, we instead generate a single sample for a batch of contexts
        noise_flow, log_probs_flow = self._flow.sample_and_log_prob(1, context=inputs) # shape = (n_batch, 1, n_discrete_dims), (n_batch, 1)
        # Squeeze out the unit dim of n_samples = 1 above
        noise_flow = torch.squeeze(noise_flow, dim=1)
        log_probs_flow = torch.squeeze(log_probs_flow, dim=1)

        # Supplement output with flow noise       
        outputs = torch.where(zero_mask, noise_flow, inputs)
 
        # Get dropout likelihoods from net
        probs_dropout = self._prob_net(outputs)
        
        # Pick out the selected probs
        log_probs_dropout_selected = torch.log(probs_dropout[torch.arange(inputs.shape[0]), prob_index])

        return outputs, log_probs_dropout_selected - log_probs_flow

    def inverse(self, inputs, context=None):
        batch_size = inputs.shape[0]

        # Probabilities to drop features
        probs_dropout = self._prob_net(inputs)
        cum_probs_dropout = torch.cumsum(probs_dropout, dim=1)

        # Select a drop probability
        larger_than_cum_probs = torch.rand(batch_size, 1) < cum_probs_dropout

        # Do the arange trick to find first nonzero
        selected_index = torch.argmax(larger_than_cum_probs*torch.arange(self._n_probs, 0, -1), axis=1)
        log_probs_dropout_selected = torch.log(probs_dropout[torch.arange(batch_size), selected_index])
        
        # Clone the input
        outputs = inputs.clone()

        # Zero out the features above the cutoff
        zero_mask = self._drop_indices > selected_index[:,None]
        outputs[zero_mask] = 0

        # Get the logprob of the flow
        log_probs_flow = self._flow.log_prob(inputs, context=outputs)

        return outputs, log_probs_flow - log_probs_dropout_selected