from typing import List

import pytest
from hypothesis import given, settings
from hypothesis.strategies import booleans
from scipy.spatial.distance import cosine

from declutr import Encoder


class TestEncoder:
    # The base model will take longer than the small model, which triggers a test timing error.
    # Turn off deadlines to avoid this.
    @settings(deadline=None)
    @given(sphereize=booleans())
    def test_encoder(self, inputs: List[str], encoder: Encoder, sphereize: bool) -> None:
        # The last two of these three sentences are most similar.
        inputs = inputs[-3:]

        # The relative ranking should not change if sphereize is True/False, so run tests with both.
        encoder._sphereize = sphereize

        # Run two distinct tests, which should cover all use cases of Encoder:
        #  1. A List[str] input where batch_size is not None.
        embeddings = encoder(inputs, batch_size=len(inputs))
        assert cosine(embeddings[0], embeddings[1]) > cosine(embeddings[1], embeddings[2])
        assert cosine(embeddings[0], embeddings[2]) > cosine(embeddings[1], embeddings[2])

        #  2. A str input where batch_size is None. Check that the expected UserWarning is raised.
        embeddings = []
        for text in inputs:
            if sphereize:
                with pytest.warns(UserWarning):
                    embeddings.append(encoder(text, batch_size=None))
            else:
                embeddings.append(encoder(text, batch_size=None))
        assert cosine(embeddings[0], embeddings[1]) > cosine(embeddings[1], embeddings[2])
        assert cosine(embeddings[0], embeddings[2]) > cosine(embeddings[1], embeddings[2])
