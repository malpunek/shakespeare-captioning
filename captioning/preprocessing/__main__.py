from .extend_verbs import main as extend_verbs
from .extract_tagged_lemmas import main as extract_lemmas
from .feature_extraction import main as extract_features
from .map_captions import main as captions_to_semantic_terms
from .plays_word_intersection import main as plays_word_intersection
from .transform_verbs import main as transform_verbs

extract_lemmas()
transform_verbs()
extend_verbs()
captions_to_semantic_terms()
plays_word_intersection()

extract_features()
