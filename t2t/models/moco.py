from typing import Dict, Optional

import torch
import torch.nn as nn

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from t2t.data.dataset_readers.dataset_utils.masked_lm_utils import mask_tokens
from t2t.losses import PyTorchMetricLearningLoss
from t2t.models.contrastive_text_encoder_util import get_anchor_positive_pairs

@Model.register("MoCo")
class MoCoText(Model):
    """
    Same as above but with added

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`
    seq2vec_encoder : `Seq2VecEncoder`
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `text_field_embedder`.
    feedforward : `FeedForward`, optional, (default = None).
        An optional feedforward layer to apply after the seq2vec_encoder.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """
    #TODO: add new args to config file
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        loss: PyTorchMetricLearningLoss,
        feedforward: Optional[FeedForward] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        dim: int = 768, # get this from config or somewhere else
        K: int = 1200, #queue size; number of negative keys (default: 65536)
        m: float = 0.999, # moco momentum of updating key encoder (default: 0.999)
        T: float = 0.001, # softmax temperature (default: 0.07)
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        # (HACK): This prevents the user from having to specify the tokenizer / masked language modeling
        # objective again. In the future it would be great to come up with something more elegant.
        self._token_embedder = self._text_field_embedder._token_embedders["tokens"]
        self._tokenizer = self._token_embedder.tokenizer
        self._masked_language_modeling = self._token_embedder.masked_language_modeling

        self._seq2vec_encoder_q = seq2vec_encoder
        self._seq2vec_encoder_k = seq2vec_encoder
        self._feedforward_q = feedforward
        self._feedforward_k = feedforward
        self.K = K
        self.m = m
        self.T = T

        if feedforward is not None:
            for param_q, param_k in zip(self._feedforward_q.parameters(), self._feedforward_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                #param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self._seq2vec_encoder_q.parameters(), self._seq2vec_encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            #param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # TODO: change this in config rather than hardcoding it here
        self._loss = nn.CrossEntropyLoss().cuda()
        initializer(self)

    #@torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self._seq2vec_encoder_q.parameters(), self._seq2vec_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        if self._feedforward_q is not None:
            for param_q, param_k in zip(self._feedforward_q.parameters(), self._feedforward_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    #@torch.no_grad() #TODO: check which functions need gradients and uncomment
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        
        # TODO: uncomment if using ddp
        #keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        if self.K % batch_size != 0:
            print('leftover batch size:', batch_size)

        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + batch_size) > self.queue.shape[1]:
            remainder = (ptr + batch_size) - self.queue.shape[1]
            self.queue[:, ptr:ptr + batch_size] = keys.T[:, :batch_size - remainder]
            self.queue[:, :remainder] = keys.T[:, batch_size - remainder:]
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(  # type: ignore
        self, tokens: TextFieldTensors,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : TextFieldTensors
            From a `TextField`

        # Returns

        An output dictionary consisting of:

        embeddings : torch.FloatTensor
            A tensor of shape `(batch_size, self._seq2vec_encoder.get_output_dim())`, which is the
            representation for the given `tokens` output by the encoder. The encoder is composed of:
            `self._text_field_embedder`, `self._seq2seq_encoder` (optional), and `self._seq2vec_encoder`, in that
            order.
        projections : torch.FloatTensor
            A tensor of shape `(batch_size, self._feedforward.get_output_dim())`, which is the non-linear
            projection of the learned representation for the given `anchor_tokens` output by the projection head.
            This field will only be included if `self._feedforward` is not `None`.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimized.
        """
        output_dict: Dict[str, torch.Tensor] = {}

        # If token_ids contains a third dimension, then spans were sampled during the data loading process.
        # sample_anchor_positive_pairs splits the batch on the second dimension to get our anchor, positive pairs.
        if tokens["tokens"]["token_ids"].dim() == 3:
            anchors, positives = get_anchor_positive_pairs(tokens)
            # Mask anchor input ids and get labels required for MLM
            if self._masked_language_modeling:
                anchors = mask_tokens(anchors, self._tokenizer)
        else:
            anchors, positives = tokens, None


        # This is the textual representation learned by a trained model that will be used for downstream tasks.
        anchor_masked_lm_loss, q = self._forward_internal(anchors, output_dict, encoder_type='Query')
        #TODO: is this normalize still necessary? taken from MoCo, try ablation
        q = nn.functional.normalize(q, dim=1)

        if positives is not None:
            #with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            #TODO: modify so that this internal call doesn't giv mlm loss
            _, k = self._forward_internal(positives, encoder_type='Key')
            k = nn.functional.normalize(k, dim=1)
            
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)

            #TODO: store as two separate losses
            output_dict["loss"] = self._loss(logits, labels)
            if anchor_masked_lm_loss is not None:
                output_dict["loss"] = output_dict["loss"] + anchor_masked_lm_loss
            

        return output_dict

    def _forward_internal(
        self, tokens: TextFieldTensors, output_dict: Optional[Dict[str, torch.Tensor]] = None, encoder_type: Optional[str] = "Query"
    ) -> torch.Tensor:
        '''
        encoder_type determines whether Query encoder or Key encoder is run
        '''

        if encoder_type == "Query":
            seq2vec_encoder = self._seq2vec_encoder_q
            feedforward = self._feedforward_q
        elif encoder_type == "Key":
            seq2vec_encoder = self._seq2vec_encoder_k
            feedforward = self._feedforward_k

        masked_lm_loss, embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        embedded_text = seq2vec_encoder(embedded_text, mask=mask)
        # Don't hold on to embeddings and projections during training.
        if output_dict is not None and not self.training:
            output_dict["embeddings"] = embedded_text.clone().detach()

        # Representations produced by the non-linear projection are used for training with a contrastive loss.
        # When embedding text with a trained model, we want the representation produced by the encoder network.
        # We therefore call these vectors "projections" to distinguish them from the "embeddings".
        # See: https://arxiv.org/abs/2002.05709
        if feedforward is not None:
            embedded_text = feedforward(embedded_text)
            if output_dict is not None and not self.training:
                output_dict["projections"] = embedded_text.clone().detach()

        return masked_lm_loss, embedded_text


# utils
# unused without ddp
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output