from typing import Tuple

import numpy as np
import torch
from overrides import overrides
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder


@Seq2VecEncoder.register("dissecting")
class DissectingEncoder(Seq2VecEncoder):
    def __init__(self, embedding_dim: int, layer_start: int = 4, context_window: int = 2):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._layer_start = layer_start
        self._context_window = context_window

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, tokens: Tuple[torch.Tensor], mask: torch.BoolTensor):
        device = tokens.device
        unmask_num = (
            np.sum(mask.cpu().numpy(), axis=1, dtype=int) - 1
        )  # Not considering the last item

        # Reshape features from list of (batch_size, seq_len, hidden_dim) for each hidden state to
        # list of (num_hidden_states, seq_len, hidden_dim) for each element in the batch. Then,
        # select the output from the `self._layer_start` layer to the last layer.
        all_layer_embedding = tokens.permute(1, 0, 2, 3)[:, self._layer_start :, :, :].cpu().numpy()

        embedding = []
        # One sentence at a time
        for sent_index in range(len(unmask_num)):
            sentence_feature = all_layer_embedding[sent_index, :, : unmask_num[sent_index], :]
            one_sentence_embedding = []
            # Process each token
            for token_index in range(sentence_feature.shape[1]):
                token_feature = sentence_feature[:, token_index, :]
                # 'Unified Word Representation'
                token_embedding = self._unify_token(token_feature)
                one_sentence_embedding.append(token_embedding)

            one_sentence_embedding = np.array(one_sentence_embedding)
            sentence_embedding = self._unify_sentence(sentence_feature, one_sentence_embedding)
            embedding.append(sentence_embedding)

        embedding = torch.as_tensor(embedding, device=device)
        return embedding

    def _unify_token(self, token_feature: torch.Tensor):
        alpha_alignment = np.zeros(token_feature.shape[0])
        alpha_novelty = np.zeros(token_feature.shape[0])

        for k in range(token_feature.shape[0]):

            left_window = token_feature[k - self._context_window : k, :]
            right_window = token_feature[k + 1 : k + self._context_window + 1, :]
            window_matrix = np.vstack([left_window, right_window, token_feature[k, :][None, :]])

            Q, R = np.linalg.qr(window_matrix.T)  # This gives negative weights

            r = R[:, -1]
            alpha_alignment[k] = np.mean(normalize(R[:-1, :-1], axis=0), axis=1).dot(R[:-1, -1]) / (
                np.linalg.norm(r[:-1])
            )
            alpha_alignment[k] = 1 / (alpha_alignment[k] * window_matrix.shape[0] * 2)
            alpha_novelty[k] = abs(r[-1]) / (np.linalg.norm(r))

        # Sum Norm
        alpha_alignment = alpha_alignment / np.sum(alpha_alignment)  # Normalization Choice
        alpha_novelty = alpha_novelty / np.sum(alpha_novelty)

        alpha = alpha_novelty + alpha_alignment

        alpha = alpha / np.sum(alpha)  # Normalize

        out_embedding = token_feature.T.dot(alpha)

        return out_embedding

    def _unify_sentence(self, sentence_feature, one_sentence_embedding):
        sent_len = one_sentence_embedding.shape[0]

        var_token = np.zeros(sent_len)
        for token_index in range(sent_len):
            token_feature = sentence_feature[:, token_index, :]
            sim_map = cosine_similarity(token_feature)
            var_token[token_index] = np.var(sim_map.diagonal(-1))

        var_token = var_token / np.sum(var_token)

        sentence_embedding = one_sentence_embedding.T.dot(var_token)

        return sentence_embedding
