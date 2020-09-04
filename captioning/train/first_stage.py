import logging
from contextlib import closing
from itertools import islice
from statistics import fmean

import torch
from nltk.translate.bleu_score import sentence_bleu
from recordclass import recordclass
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

from ..config import (
    coco_train_conf,
    coco_val_conf,
    device,
    experiment_folder,
    first_stage,
    shakespare_conf,
)
from ..dataset import ValidationDataset, QuickCocoDataset
from ..model import TermDecoder
from ..utils import get_yn_response


def extract_caption_len(captions):
    captions_lens = captions[:, -1]
    captions = captions[:, :-1]
    return captions, captions_lens


def to_batch_format(sample):
    feats, caption = sample
    feats, caption = torch.Tensor(feats), caption
    feats = feats.unsqueeze(0)
    caption = caption.unsqueeze(0)
    caption, caption_len = extract_caption_len(caption)
    return feats, caption, caption_len


def train(dataset, mapping, model, writer, criterion, optimizer):

    dataloader = DataLoader(
        dataset, batch_size=first_stage["batch_size"], num_workers=4, shuffle=True
    )
    sample_feats, sample_caption, sample_caption_len = to_batch_format(dataset[0])

    model = model.train().to("cpu")
    writer.add_graph(
        model, input_to_model=(sample_feats, sample_caption, sample_caption_len)
    )
    model = model.to(device)

    for epoch in trange(first_stage["epochs"], desc="Epochs"):
        running_loss = 0
        model = model.train()

        for i, data in enumerate(tqdm(dataloader, desc="Batches")):
            features, captions = data

            captions, caption_lens = extract_caption_len(captions)

            caption_lens += 1  # We add the <start> token

            captions = captions.to(device)
            features = features.to(device)
            caption_lens = caption_lens.to(device)

            optimizer.zero_grad()

            targets = captions.detach().clone()[:, 1:]
            outputs, hidden = model(
                features, captions[:, :-1].detach().clone(), caption_lens
            )
            loss = criterion(outputs.permute(0, 2, 1), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 50 == 49:
                step_number = epoch * len(dataloader) + i
                writer.add_scalar("Training loss", running_loss / 50, step_number)

                running_loss = 0

                model.eval()
                words, confidence = model.forward_eval(sample_feats.to(device), mapping)

                tmp = sample_caption.reshape((-1)).tolist()
                writer.add_text("Target", f"{list(mapping.decode(tmp))}", step_number)
                writer.add_text("Predictions", f"{words}", step_number)
                writer.add_scalar(
                    "Mean confidence", sum(confidence) / len(confidence), step_number
                )

                model.train()

        # after each epoch
        yield model


def precision(target, prediction):
    """Precision = TP/(TP + FP)"""
    if len(prediction) == 0:
        return 0
    tp = sum(1 for p in prediction if p in target)
    return tp / len(prediction)


def recall(target, prediction):
    """Recall = TP/(TP + FN)"""
    if len(target) == 0:
        logging.warning("target shouldnt be empty (first_stage:train:recall)")
        return 0
    tp = sum(1 for p in prediction if p in target)
    return tp / len(target)


Score = recordclass(
    "Score",
    ["bleu", "avg_precision", "max_precision", "avg_recall", "max_recall"],
    defaults=[0, 0, 0, 0, 0],
)


def evaluate(model, mapping):

    if not hasattr(evaluate, "dataset"):
        evaluate.dataset = ValidationDataset(
            coco_val_conf["features"],
            coco_val_conf["final"],
        )
    else:
        evaluate.dataset.open_feats()

    model.eval()
    model.to(device)

    score = Score()

    eval_size = 1000

    for feats, targets in tqdm(
        islice(evaluate.dataset, eval_size), total=eval_size, desc="Evaluating"
    ):
        feats = torch.Tensor(feats).unsqueeze(0)
        prediction, confidence = model.forward_eval(feats.to(device), mapping)
        prediction = prediction[1:-1]  # strip <start> and <end>

        score.bleu += sentence_bleu(targets, prediction) / eval_size

        p = list(precision(t, prediction) for t in targets)
        r = list(recall(t, prediction) for t in targets)

        score.avg_precision += fmean(p) / eval_size
        score.max_precision += max(p) / eval_size
        score.avg_recall += fmean(r) / eval_size
        score.max_recall += max(r) / eval_size

    evaluate.dataset.close()
    return score


def main():
    dataset = QuickCocoDataset(
        coco_train_conf["features"], coco_train_conf["final"], shakespare_conf["final"],
    )

    mapping = dataset.get_term_mapping

    writer = SummaryWriter(experiment_folder)

    vocab_size = len(mapping)
    # TODO: config
    model = TermDecoder(vocab_size, 2048, 2048)

    print(f"SCORE: {evaluate(model, mapping)}")

    criterion = nn.NLLLoss()  # TODO try nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=first_stage["learning_rate"])

    with closing(dataset):
        try:
            for epoch, trained_model in enumerate(
                train(dataset, mapping, model, writer, criterion, optimizer)
            ):
                torch.save(
                    trained_model.state_dict(),
                    experiment_folder / f"model_ep{(epoch + 1):03d}.pth",
                )
                score = evaluate(trained_model, mapping)
                step = (epoch + 1) * len(dataset) // first_stage["batch_size"]

                for name, value in score._asdict().items():
                    writer.add_scalar(f"Score: {name}", value, step)

        except:  # noqa
            if get_yn_response("Remove experiment folder? [y/N]"):
                import shutil

                shutil.rmtree(experiment_folder, ignore_errors=True)
            raise

    writer.close()


if __name__ == "__main__":
    main()
