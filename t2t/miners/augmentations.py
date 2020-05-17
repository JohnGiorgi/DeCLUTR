from allennlp.common import Registrable
import nlpaug.augmenter.word as naw


class NLPAug(Registrable):
    """"""
    pass


@NLPAug.register("contextual-word-embed-aug")
class ContextualWordEmbsAug(NLPAug, naw.ContextualWordEmbsAug)
    def __init__(
        self, # Same things as ContextualWordEmbsAug
    ) -> None:

        super().__init__(
            # same things as ContextualWordEmbsAug
        ) 