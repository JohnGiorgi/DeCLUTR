from allennlp.common import Registrable
import nlpaug.augmenter.word as naw
import os
os.environ["MODEL_DIR"] = '../model'

class NLPAug(Registrable):
    """"""
    pass


@NLPAug.register("contextual-word-embed-aug")
class ContextualWordEmbsAug(NLPAug, naw.ContextualWordEmbsAug):
    def __init__(
        self,
        model_path='bert-base-uncased',
        action="substitute",
        device='cpu'
    ) -> None:
        super().__init__(
            model_path=model_path,
            action=action,
            device=device
        ) 
        
@NLPAug.register("synonym-aug")
class ContextualWordEmbsAug(NLPAug, naw.SynonymAug):
    def __init__(
        self,
        aug_src='wordnet'
    ) -> None:
        super().__init__(
            aug_src=aug_src
        ) 