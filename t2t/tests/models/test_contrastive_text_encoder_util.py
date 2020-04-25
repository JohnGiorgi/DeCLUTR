from t2t.models.contrastive_text_encoder_util import get_anchor_positive_pairs
from allennlp.data import TextFieldTensors
import torch


class TestContrastiveTextEncoderUtil:
    def test_get_anchor_positive_pairs(self):
        batch_size, max_len = 4, 12  # arbitrary

        # Create some dummy data
        anchors_expected = torch.randn(batch_size, 1, max_len)
        positives_expected = torch.randn(batch_size, 1, max_len)
        token_ids = torch.cat((anchors_expected, positives_expected), dim=1)
        mask = token_ids.clone()
        type_ids = token_ids.clone()

        tokens: TextFieldTensors = {
            "tokens": {"token_ids": token_ids, "mask": mask, "type_ids": type_ids}
        }

        anchors_actual, positives_actual = get_anchor_positive_pairs(tokens)

        # Check that the function properly splits up the input TextFieldTensors
        for tensor in anchors_actual["tokens"].values():
            assert torch.equal(tensor, anchors_expected.squeeze(1))
        for tensor in positives_actual["tokens"].values():
            assert torch.equal(tensor, positives_expected.squeeze(1))
