from typing import Tuple

import torch
from transformers import PreTrainedTokenizer

from allennlp.data import TextFieldTensors


def _mask_tokens(
    inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10%
    original. Copied from:
    https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py"""

    if tokenizer.mask_token is None:
        raise ValueError(
            (
                "This tokenizer does not have a mask token which is necessary for masked language"
                " modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability
    # mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def mask_tokens(
    tokens: TextFieldTensors,
    tokenizer: PreTrainedTokenizer,
    mlm_probability: float = 0.15,
) -> TextFieldTensors:
    device = tokens["tokens"]["token_ids"].device
    inputs, labels = _mask_tokens(
        inputs=tokens["tokens"]["token_ids"].to("cpu"),
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
    )
    tokens["tokens"]["token_ids"] = inputs.to(device)
    tokens["tokens"]["masked_lm_labels"] = labels.to(device)
    return tokens
