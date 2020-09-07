def filter_short(caps, min_len=3):
    return len(caps["terms"]) > min_len


def extract_caption_len(captions):
    captions_lens = captions[:, -1]
    captions = captions[:, :-1]
    return captions, captions_lens
