def filter_fn(caps):
    return len(caps["terms"]) > 3


def extract_caption_len(captions):
    captions_lens = captions[:, -1]
    captions = captions[:, :-1]
    return captions, captions_lens
