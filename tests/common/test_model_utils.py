from copy import deepcopy

import torch
from allennlp.data import TextFieldTensors
from hypothesis import given, settings
from hypothesis.strategies import integers

from declutr.common import model_utils


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
        actual_three_input_dim = model_utils.unpack_batch(deepcopy(three_dim_input))
        for name, tensor in actual_three_input_dim["tokens"].items():
            assert torch.equal(
                tensor,
                three_dim_input["tokens"][name].reshape(batch_size * num_anchors, max_length),
            )
        # ...unpack_batch is a no-op for TextFieldTensors with tensors less than or greater than 3D.
        actual_two_dim_input = model_utils.unpack_batch(deepcopy(two_dim_input))
        for name, tensor in actual_two_dim_input["tokens"].items():
            assert torch.equal(tensor, two_dim_input["tokens"][name])
        actual_four_dim_input = model_utils.unpack_batch(deepcopy(four_dim_input))
        for name, tensor in actual_four_dim_input["tokens"].items():
            assert torch.equal(tensor, four_dim_input["tokens"][name])

    def test_all_gather_anchor_positive_pairs_no_op(self) -> None:
        """Check that `all_gather_anchor_positive_pairs` is a no-op when not in distributed mode."""
        num_anchors = 2
        num_positives = 2
        batch_size = 16
        embedding_dim = 256

        expected_anchors = torch.randn(num_anchors, batch_size, embedding_dim)
        expected_positives = torch.randn(num_positives, batch_size, embedding_dim)
        actual_anchors, actual_positives = model_utils.all_gather_anchor_positive_pairs(
            expected_anchors, expected_positives
        )

        assert torch.equal(actual_anchors, expected_anchors)
        assert torch.equal(actual_positives, expected_positives)
