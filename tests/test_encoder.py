from pathlib import Path
from typing import List

import pytest
import torch
from declutr import Encoder
from hypothesis import given, settings
from hypothesis.strategies import booleans
from torch.nn import CosineSimilarity


class TestEncoder:
    cosine = CosineSimilarity(dim=-1)

    # The base model will take longer than the small model, which triggers a test timing error.
    # Turn off deadlines to avoid this.
    @settings(deadline=None)
    @given(sphereize=booleans())
    def test_encoder(
        self, inputs: List[str], inputs_filepath: Path, encoder: Encoder, sphereize: bool
    ) -> None:
        # The relative ranking should not change if sphereize is True/False, so run tests with both.
        encoder._sphereize = sphereize

        # Run three distinct tests, which should cover all use cases of Encoder:
        #  1. A List[str] input where batch_size is not None.
        embeddings = encoder(inputs, batch_size=len(inputs))
        embeddings = torch.from_numpy(embeddings)
        # These are hard-coded examples that should have the highest cosine similarity.
        assert torch.topk(self.cosine(embeddings[2], embeddings), k=2)[-1][-1].item() == 3
        assert torch.topk(self.cosine(embeddings[6], embeddings), k=2)[-1][-1].item() == 7

        #  2. A str input where batch_size is None. Check that the expected UserWarning is raised.
        embeddings = []
        for text in inputs:
            if sphereize:
                with pytest.warns(UserWarning):
                    embeddings.append(encoder(text, batch_size=None))
            else:
                embeddings.append(encoder(text, batch_size=None))
        embeddings = torch.as_tensor(embeddings).squeeze(1)
        assert torch.topk(self.cosine(embeddings[2], embeddings), k=2)[-1][-1].item() == 3
        assert torch.topk(self.cosine(embeddings[6], embeddings), k=2)[-1][-1].item() == 7

        #  3. A filepath input that points to file with one example per line.
        embeddings = encoder(inputs_filepath, batch_size=len(inputs))
        embeddings = torch.from_numpy(embeddings)
        assert torch.topk(self.cosine(embeddings[2], embeddings), k=2)[-1][-1].item() == 3
        assert torch.topk(self.cosine(embeddings[6], embeddings), k=2)[-1][-1].item() == 7
