from pathlib import Path
from xml.etree import ElementTree as ET
import json

from nltk import tokenize
import ebooklib
from ebooklib import epub

from .to_frames import cap_to_ascii, match
from .to_final import reduce_frames, get_frame_mapping, extract_frames
from ..config import tolkien_fldr, tolkien_conf, coco_train_conf
from ..utils import ask_overwrite


w3c = "{http://www.w3.org/1999/xhtml}"


def xml_to_text(x):
    root = x
    if root[1].tag != f"{w3c}body" or root[1][0].tag != f"{w3c}div":
        print("Error")
        assert False

    chapter = root[1][0]
    for t in (f"{w3c}a", f"{w3c}h1"):
        junk = chapter.findall(t)
        for j in junk:
            chapter.remove(j)

    text = [t.strip() for t in chapter.itertext()]
    return " ".join(text)


def get_book_text(book_path):
    book = epub.read_epub(book_path)
    chapters = [
        doc
        for doc in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
        if "chap" in str(doc)
    ]
    chapters = [str(c.content, "utf-8") for c in chapters]
    chapters = list(map(ET.fromstring, chapters))

    return " ".join([xml_to_text(x) for x in chapters])


def to_txt():
    if not ask_overwrite(tolkien_conf["txt"]):
        return

    books = Path(tolkien_fldr).iterdir()

    tolk = " ".join([get_book_text(b) for b in books if b.suffix == ".epub"])
    tolk = " ".join(tolk.split())
    tolk = cap_to_ascii(tolk)
    sents = tokenize.sent_tokenize(tolk)

    with open(tolkien_conf["txt"], "wt") as f:
        for sent in filter(lambda s: len(s.split()) > 5, sents):
            f.write(f"{sent}\n")

    with open(tolkien_conf["txt"]) as in_f, open(tolkien_conf["basic"], "wt") as out_f:
        basic = [{"caption": line.strip()} for line in in_f.readlines()]
        json.dump(basic, out_f, indent=2)


def main():
    to_txt()
    match(tolkien_conf["conll"], tolkien_conf["basic"], tolkien_conf["frames"])

    if not ask_overwrite(tolkien_conf["final"]):
        return

    with open(coco_train_conf["frames"]) as cf:
        coco = json.load(cf)

    fmap = get_frame_mapping(extract_frames(coco))
    fmap = {f"{key}_FRAME": f"{val}_FRAME" for key, val in fmap.items()}

    reduce_frames(tolkien_conf["frames"], tolkien_conf["final"], fmap)


if __name__ == "__main__":
    main()
