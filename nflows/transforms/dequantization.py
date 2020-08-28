import torch

from nflows.transforms.base import Transform
from nflows.utils import torchutils

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

        return outputs
        
    def inverse(self, inputs, context=None):
        # Check if the final dims of inputs correspond with the mask
        batch_size = inputs.shape[0]
        
        # Expand mask and max_labels with the batch dimension
        batched_mask       = self._mask.expand(batch_size, *self._mask.shape)
        batched_max_labels = self._max_labels.repeat(batch_size)

        # Scale to original label and floor
        outputs = inputs.clone()
        outputs[batched_mask] = torch.floor(outputs[batched_mask]*batched_max_labels)

        return outputs

def main():
    print("What")


if __name__ == "__main__":
    main()