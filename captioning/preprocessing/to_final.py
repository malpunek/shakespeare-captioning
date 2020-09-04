import json
import re
from collections import Counter
from itertools import chain
from operator import itemgetter

import networkx as nx
from tqdm.auto import tqdm

from ..config import coco_train_conf, coco_val_conf, load_framenet, shakespare_conf
from ..utils import ask_overwrite

fn = load_framenet()


def to_words(caption):
    caption = re.sub(r"\W+", " ", caption)
    return caption.lower().split()


def extract_frames(annotations):
    res = map(itemgetter("terms"), annotations)
    res = chain.from_iterable(res)
    res = filter(lambda x: x.endswith("_FRAME"), res)
    res = map(lambda x: x[:-6], res)
    # res = map(lambda x: fn.frame_by_name(x), res)
    return res


def get_framenet_graph():

    # create graph nodes
    dg = nx.DiGraph()
    for f in fn.frames():
        dg.add_node(f.name)

    # add edges
    relations = ["Inheritance", "Subframe", "Using"]
    for f in fn.frames():
        for fr in f["frameRelations"]:
            if fr["type"]["name"] in relations and fr["superFrameName"] != f.name:
                dg.add_edge(
                    f.name, fr["superFrameName"], key=None, edge_type=fr["type"]["name"]
                )

    return dg


def annotate_framenet_graph(graph, coco_counts):
    for node in graph.nodes:
        graph.nodes[node]["count"] = 0
    for frame_name, occurances in coco_counts.items():
        graph.nodes[frame_name]["count"] = occurances
    return graph


# Copied from
# https://github.com/computationalmedia/semstyle/blob/master/original_code/scripts/framenet_verbs_better_compress.ipynb
def compress_framenet_graph(frame_graph, th=200):
    parent_graph = nx.DiGraph()
    parent_graph.add_nodes_from(frame_graph.nodes(data=True))

    sp = list(nx.all_pairs_shortest_path_length(frame_graph))
    for f, parents in sp:

        # dont compress nodes which have a high count
        if frame_graph.nodes[f]["count"] >= th:
            parent_graph.add_edge(f, f)
            continue

        # find the closest parent over threshold
        best_dist = 9999
        best_p = None
        for p, d in parents.items():
            if frame_graph.nodes[p]["count"] >= th:
                if d < best_dist:
                    best_dist = d
                    best_p = p

        # no parents above threshold
        if best_p is None:
            # get the highest parent
            best_p, best_dist = sorted(parents.items(), key=lambda x: -x[1])[0]

        # compress edges
        parent_graph.add_edge(f, best_p)
    return parent_graph


def get_frame_mapping(coco_frames):
    """Should return map(Frame.name, Frame.name).
    All keys should be replaced by values in the final version"""
    g = get_framenet_graph()
    g = annotate_framenet_graph(g, Counter(coco_frames))
    g = compress_framenet_graph(g, th=20)
    return {p: q for p, q in list(g.edges)}


def new_terms(fmap, terms):
    return [(term if term not in fmap else fmap[term]) for term in terms]


def reduce_frames(frame_path, out_path, fmap):
    if not ask_overwrite(out_path):
        return

    with open(frame_path) as f:
        anns = json.load(f)

    new_anns = [
        {
            **ann,
            "caption_words": to_words(ann["caption"]),
            "terms": new_terms(fmap, ann["terms"]),
        }
        for ann in anns
    ]

    if len(new_anns) > 0 and "original" in new_anns[0]:
        new_anns = [
            {**ann, "original_words": to_words(ann["original"])} for ann in new_anns
        ]

    with open(out_path, "wt") as f:
        json.dump(new_anns, f, indent=2)


def main():
    with open(coco_train_conf["frames"]) as cf:
        coco = json.load(cf)

    fmap = get_frame_mapping(extract_frames(coco))
    fmap = {f"{key}_FRAME": f"{val}_FRAME" for key, val in fmap.items()}

    for conf in tqdm(
        (coco_train_conf, coco_val_conf, shakespare_conf),
        desc="Mapping to reduced frame vocabulary",
    ):
        reduce_frames(conf["frames"], conf["final"], fmap)


if __name__ == "__main__":
    main()
