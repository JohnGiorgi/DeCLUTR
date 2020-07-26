from copy import deepcopy

import torch
from allennlp.data import TextFieldTensors
from hypothesis import given, settings
from hypothesis.strategies import integers

from declutr.common.model_utils import unpack_batch


class TestModelUtils:
    @settings(deadline=None)
    @given(
        batch_size=integers(min_value=1, max_value=4),
        num_anchors=integers(min_value=1, max_value=4),
        max_length=integers(min_value=1, max_value=16),
    )
    def test_unpack_batch(self, batch_size: int, num_anchors: int, max_length: int) -> None:
        # Create some dummy data.
        two_dim_tensor = torch.randn(batch_size, max_length)
        two_dim_input: TextFieldTensors = {
            "tokens": {
                "token_ids": two_dim_tensor,
                "mask": torch.ones_like(two_dim_tensor),
                "type_ids": torch.ones_like(two_dim_tensor),
            }
        }
        three_dim_tensor = torch.randn(batch_size, num_anchors, max_length)
        three_dim_input: TextFieldTensors = {
            "tokens": {
                "token_ids": three_dim_tensor,
                "mask": torch.ones_like(three_dim_tensor),
                "type_ids": torch.ones_like(three_dim_tensor),
            }
        }
        four_dim_tensor = torch.randn(batch_size, num_anchors, num_anchors, max_length)
        four_dim_input: TextFieldTensors = {
            "tokens": {
                "token_ids": four_dim_tensor,
                "mask": torch.ones_like(four_dim_tensor),
                "type_ids": torch.ones_like(four_dim_tensor),
            }
        }

        # Only TextFieldTensors with tensors of three dimensions should be reshaped...
        # Tensors are updated in-place, so deepcopy before passing to unpack_batch
        actual_three_input_dim = unpack_batch(deepcopy(three_dim_input))
        for name, tensor in actual_three_input_dim["tokens"].items():
            assert torch.equal(
                tensor,
                three_dim_input["tokens"][name].reshape(batch_size * num_anchors, max_length),
            )
        # ...unpack_batch is a no-op for TextFieldTensors with tensors less than or greater than 3D.
        actual_two_dim_input = unpack_batch(deepcopy(two_dim_input))
        for name, tensor in actual_two_dim_input["tokens"].items():
            assert torch.equal(tensor, two_dim_input["tokens"][name])
        actual_four_dim_input = unpack_batch(deepcopy(four_dim_input))
        for name, tensor in actual_four_dim_input["tokens"].items():
            assert torch.equal(tensor, four_dim_input["tokens"][name])
