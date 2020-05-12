import numpy as np
import torch
from hypothesis import given
from hypothesis.strategies import integers

from allennlp.data import TextFieldTensors
from t2t.models.contrastive_text_encoder_util import chunk_positives


class TestContrastiveTextEncoderUtil:
    @given(
        batch_size=integers(min_value=1, max_value=3),
        num_positives=integers(min_value=1, max_value=3),
    )
    def test_chunk_positives(self, batch_size: int, num_positives: int):
        # Create some dummy data
        max_len = 12  # arbitrary, pick a small value so tests are fast
        chunks_expected = torch.randn(batch_size, num_positives, max_len)
        token_ids = chunks_expected.clone()
        mask = chunks_expected.clone()
        type_ids = chunks_expected.clone()

        chunk_dim = np.argmin([batch_size, num_positives]).item()
        chunk_size = chunks_expected.size(chunk_dim)

        tokens: TextFieldTensors = {
            "tokens": {"token_ids": token_ids, "mask": mask, "type_ids": type_ids}
        }

        chunks_actual = chunk_positives(tokens, chunk_dim=chunk_dim)

        # Check that the number of chunks and their sizes are as expected
        assert len(chunks_actual) == chunk_size
        for chunk in chunks_actual:
            for tensor in chunk["tokens"].values():
                assert tensor.size() == (max(batch_size, num_positives), max_len)
