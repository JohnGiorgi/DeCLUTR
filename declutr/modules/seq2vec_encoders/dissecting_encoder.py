import torch
from overrides import overrides

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from declutr.common import cosine_similarity, normalize


@Seq2VecEncoder.register("dissecting")
class DissectingEncoder(Seq2VecEncoder):
    """The "dissecting" pooler introduced in SBERT-WK: https://arxiv.org/abs/2002.06652.

    Code adapted from: https://github.com/BinWang28/SBERT-WK-Sentence-Embedding.
    """

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

    @torch.no_grad()
    def forward(self, tokens: torch.tensor, mask: torch.BoolTensor) -> torch.Tensor:
        unmask_num = (
            torch.sum(mask, dim=1, dtype=torch.int).cpu() - 1
        )  # Not considering the last item

        # Reshape features from (num_hidden_states, batch_size, seq_len, hidden_dim) to
        # (batch_size, num_hidden_states, seq_len, hidden_dim). Then, select the output from the
        # `self._layer_start` layer to the last layer.
        all_layer_embedding = tokens.permute(1, 0, 2, 3)[:, self._layer_start :, :, :].cpu()

        # One sentence at a time
        embedding = []
        for sent_index in range(len(unmask_num)):
            sentence_feature = all_layer_embedding[sent_index, :, : unmask_num[sent_index], :]
            one_sentence_embedding = []
            # Process each token
            for token_index in range(sentence_feature.size(1)):
                token_feature = sentence_feature[:, token_index, :]
                # 'Unified Word Representation'
                token_embedding = self._unify_token(token_feature)
                one_sentence_embedding.append(token_embedding)

            one_sentence_embedding = torch.stack(one_sentence_embedding)
            sentence_embedding = self._unify_sentence(sentence_feature, one_sentence_embedding)
            embedding.append(sentence_embedding)

        embedding = torch.stack(embedding)
        embedding = embedding.to(tokens.device)

        return embedding

    @torch.no_grad()
    def _unify_token(self, token_feature: torch.Tensor) -> torch.Tensor:
        alpha_alignment = torch.zeros(token_feature.size(0))
        alpha_novelty = torch.zeros(token_feature.size(0))

        for k in range(token_feature.size(0)):

            left_window = token_feature[k - self._context_window : k, :]
            right_window = token_feature[k + 1 : k + self._context_window + 1, :]
            window_matrix = torch.cat([left_window, right_window, token_feature[k, :][None, :]])

            _, R = torch.qr(window_matrix.t())  # This gives negative weights

            r = R[:, -1]
            alpha_alignment[k] = (
                torch.mean(normalize(R[:-1, :-1], dim=0), dim=1) @ R[:-1, -1] / (torch.norm(r[:-1]))
            )
            alpha_alignment[k] = 1 / (alpha_alignment[k] * window_matrix.size(0) * 2)
            alpha_novelty[k] = abs(r[-1]) / (torch.norm(r))

        # Sum Norm
        alpha_alignment = alpha_alignment / torch.sum(alpha_alignment)  # Normalization Choice
        alpha_novelty = alpha_novelty / torch.sum(alpha_novelty)

        alpha = alpha_novelty + alpha_alignment
        alpha = alpha / torch.sum(alpha)  # Normalize

        out_embedding = token_feature.t() @ alpha

        return out_embedding

    @torch.no_grad()
    def _unify_sentence(
        self, sentence_feature: torch.Tensor, one_sentence_embedding: torch.Tensor
    ) -> torch.Tensor:
        sent_len = one_sentence_embedding.size(0)
        var_token = torch.zeros(sent_len)

        for token_index in range(sent_len):
            token_feature = sentence_feature[:, token_index, :]
            sim_map = cosine_similarity(token_feature)
            var_token[token_index] = torch.var(sim_map.diagonal(-1))

        var_token = var_token / torch.sum(var_token)
        sentence_embedding = one_sentence_embedding.t() @ var_token

        return sentence_embedding
