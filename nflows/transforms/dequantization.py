import torch

from nflows.transforms.base import Transform
from nflows.utils import torchutils

from torch.nn import functional as F
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.splines import rational_quadratic
from nflows.transforms.permutations import RandomPermutation
from nflows.distributions.uniform import BoxUniform
from nflows.transforms.base import CompositeTransform
from nflows.flows.base import Flow


class UniformDequantization(Transform):
    '''
    A transform that uniformly dequantizes discrete dimensions

    Initialized with a tensor that contains the maximum discrete labels
    Labels are assumed to start at 0
    Continuous dimensions are indicated with a -1
    '''

    def __init__(self, max_labels):
        super(UniformDequantization, self).__init__()

        # Mask used to project out the discrete dimensions
        # shape = (n_batch, data_dim_1, ..., data_dim_n)
        self._mask = max_labels > 0

        # Add one because dequantization adds uniform noise
        # shape = (n_batch, n_discrete_dims)
        self._max_labels = max_labels[self._mask] + 1

        self._shape = max_labels.shape

    def forward(self, inputs, context=None):
        # Check if the final dims of inputs correspond with the mask
        batch_size = inputs.shape[0]

        # Expand mask and max_labels with the batch dimension
        batched_mask       = self._mask.expand(batch_size, *self._mask.shape)
        batched_max_labels = self._max_labels.repeat(batch_size)

        # Sample noise in the shape of batched_max_labels
        noise = torch.rand(batched_max_labels.shape)

        # Add noise to discrete dimensions and normalize
        outputs = inputs.clone()
        outputs[batched_mask] = (outputs[batched_mask] + noise) / batched_max_labels

        return outputs, torch.zeros(batch_size)
        
    def inverse(self, inputs, context=None):
        # Check if the final dims of inputs correspond with the mask
        batch_size = inputs.shape[0]
        
        # Expand mask and max_labels with the batch dimension
        batched_mask       = self._mask.expand(batch_size, *self._mask.shape)
        batched_max_labels = self._max_labels.repeat(batch_size)

        # Scale to original label and floor
        outputs = inputs.clone()
        outputs[batched_mask] = torch.floor(outputs[batched_mask]*batched_max_labels)

        return outputs, torch.zeros(batch_size)

class VariationalDequantization(Transform):
    '''
    A transform that variationally dequantizes discrete dimensions

    Initialized with a tensor that contains the maximum discrete labels
    Labels are assumed to start at 0
    Continuous dimensions are indicated with a -1
    '''
    def __init__(self, max_labels,
            rqs_hidden_features,
            rqs_flow_layers=5,
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
            rqs_min_derivative = rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        ):
        super(VariationalDequantization, self).__init__()

        # Mask used to project out the discrete dimensions
        self._mask = max_labels > 0 # shape = (dim_data)

        # Add one because dequantization adds uniform noise
        self._max_labels = max_labels[self._mask] + 1 # shape = (n_discrete_dims)

        
        self._shape = max_labels.shape[0] # shape = 1        
        self._masked_shape = self._max_labels.shape[0] # shape = 1

        # Set up a flow 
        base_dist = BoxUniform(torch.zeros(self._masked_shape), torch.ones(self._masked_shape))
        transforms = []
        for _ in range(rqs_flow_layers):
            transforms.append(RandomPermutation(features=self._masked_shape))
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
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
            ))
        transform = CompositeTransform(transforms)
        self._flow = Flow(transform, base_dist)

    def forward(self, inputs, context=None):
        # Check if the final dims of inputs correspond with the mask
        batch_size = inputs.shape[0]

        # Expand mask and max_labels with the batch dimension
        batched_mask       = self._mask.expand(batch_size, *self._mask.shape) # shape = (n_batch, n_discrete_dims)
        batched_max_labels = self._max_labels.repeat(batch_size) # shape = (n_batch * n_discrete_dims)

        # Sample noise in the shape of batched_max_labels
        # Note - we have moved the batch size of the sample into the context. 
        # i.e rather than generating a sample for every context, we instead generate a single sample for a batch of contexts
        # We're essentially quite lucky that this is possible, because in the inverse direction 
        # one has to evaluate the llh for all samples and all context of all of these samples, while only the diagonal entries are kept
        noise, logprob = self._flow.sample_and_log_prob(1, context=inputs) # shape = (n_batch, 1, n_discrete_dims), (n_batch, 1)

        # Reshape to a 1d tensor
        noise = torch.reshape(noise, (-1,)) # shape = (n_batch * n_discrete_dims)
        
        # Add noise to discrete dimensions and normalize
        outputs = inputs.clone() # shape = (n_batch, dim_data)
        outputs[batched_mask] = (outputs[batched_mask] + noise) / batched_max_labels # shape = (n_batch, dim_data)

        return outputs, -logprob

    def inverse(self, inputs, context=None):
        # Check if the final dims of inputs correspond with the mask
        batch_size = inputs.shape[0]
        
        # Expand mask and max_labels with the batch dimension
        batched_mask       = self._mask.expand(batch_size, *self._mask.shape) # shape = (n_batch, n_discrete_dims)
        batched_max_labels = self._max_labels.repeat(batch_size) # shape = (n_batch * n_discrete_dims)

        # Scale to original label and floor
        outputs = inputs.clone()
        outputs[batched_mask] = torch.floor(outputs[batched_mask]*batched_max_labels) # shape = (n_batch, dim_data)

        # Compute the logprob
        inputs_masked_reshaped = torch.reshape(inputs[batched_mask], (batch_size, -1)) # shape = (n_batch, n_discrete_dims)
        
        print(inputs_masked_reshaped)
        print(outputs.shape)
        logprob = torch.reshape(self._flow.log_prob(inputs_masked_reshaped, context=outputs), (batch_size, 1))

        return outputs, logprob


def main():
    print("What")


if __name__ == "__main__":
    main()