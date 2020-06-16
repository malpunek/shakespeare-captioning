from .extend_verbs import main as extend_verbs
from .extract_tagged_lemmas import main as extract_lemmas
from .transform_verbs import main as transform_verbs
from .feature_extraction import main as extract_features

extract_lemmas()
transform_verbs()
extend_verbs()

extract_features()
