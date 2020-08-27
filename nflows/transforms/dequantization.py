import numpy as np
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
        self._mask = max_labels > 0
        # Add one such that
        # - discrete dimensions can be divided by _max_labels to give a number between 0 and 1
        # - continuous dimensions have a 1 such that dividing doesn't do anything
        self._max_labels = torch.where(max_labels > 0, max_labels, torch.zeros_like(max_labels)) + 1

        self._shape = max_labels.shape

    def forward(self, inputs, context=None):
        # Check if the final dims of inputs correspond with the mask
        batch_size = inputs.shape[0]
        
        # Expand mask and max_labels with the batch dimension
        batched_max_labels = self._max_labels.expand(batch_size, *self._max_labels.shape)
        batched_mask       = self._mask.expand(batch_size, *self._mask.shape)
        
        # Project out the continuous variables
        inputs_masked = torch.where(batched_mask, inputs, torch.zeros_like(inputs))

        # Sample noise and rescale to [0,1]
        samples_masked = torch.where(batched_mask, torch.rand(inputs.shape), torch.zeros_like(inputs))
        outputs_masked = (inputs_masked + samples_masked)/batched_max_labels

        # Add back the continuous dimensions
        return outputs_masked + torch.where(self._mask, torch.zeros_like(inputs), inputs)

    def inverse(self, inputs, context=None):
        # Check if the final dims of inputs correspond with the mask
        batch_size = inputs.shape[0]
        
        # Expand mask and max_labels with the batch dimension
        batched_max_labels = self._max_labels.expand(batch_size, *self._max_labels.shape)
        batched_mask       = self._mask.expand(batch_size, *self._mask.shape)

        # Project out the continuous variables
        inputs_masked = torch.where(batched_mask, inputs, torch.zeros_like(inputs))

        # Scale to label and floor
        outputs_masked = torch.floor(inputs_masked*batched_max_labels)

        # Add back the continuous dimensions
        return outputs_masked + torch.where(batched_mask, torch.zeros_like(inputs), inputs)