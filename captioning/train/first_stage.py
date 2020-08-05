import json
from contextlib import closing

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm, trange

from ..config import (
    device,
    extended_word_map_path,
    first_stage,
    coco_train_conf,
    experiment_folder,
)
from ..dataset import CaptionHdf5Dataset
from ..model import TermDecoder
from ..utils import WordIdxMap, get_yn_response


def get_sample_data(dataset):
    feats, caption = dataset[0]
    feats, caption = torch.Tensor(feats), torch.Tensor(caption).to(torch.long)
    feats = feats.unsqueeze(0)
    caption = caption.unsqueeze(0)
    caption_len = caption[:, -1]
    caption = caption[:, :-1]
    return feats, caption, caption_len


def _main(dataset, mapping, vocab_size):

    writer = SummaryWriter(experiment_folder)

    dataloader = DataLoader(
        dataset, batch_size=first_stage["batch_size"], num_workers=4, shuffle=True
    )
    sample_feats, sample_caption, sample_caption_len = get_sample_data(dataset)

    # TODO: config
    model = TermDecoder(vocab_size, 2048, 2048)
    model = model.train()
    writer.add_graph(
        model, input_to_model=(sample_feats, sample_caption, sample_caption_len)
    )
    model = model.to(device)

    criterion = nn.NLLLoss()  # TODO try nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=first_stage["learning_rate"])

    # TODO tqdm
    for epoch in trange(first_stage["epochs"], desc="Epochs"):
        running_loss = 0

        for i, data in enumerate(tqdm(dataloader, desc="Batches")):
            features, captions = data

            caption_lens = captions[:, -1] + 1  # We add the <start> token
            captions = captions[:, :-1]

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
                writer.add_text(
                    "Target", f"{mapping.decode(sample_caption.reshape((-1)).tolist())}"
                )
                writer.add_text("Predictions", f"{words}")
                writer.add_scalar(
                    "Mean confidence", sum(confidence) / len(confidence), step_number
                )

                model.train()

        # after each epoch
        torch.save(model.state_dict, experiment_folder / f"model_ep{epoch:03d}.pth")

    writer.close()


def main():
    dataset = CaptionHdf5Dataset(
        coco_train_conf["features_path"], coco_train_conf["transformed_data_path"]
    )

    with open(extended_word_map_path) as f:
        word_map = json.load(f)

    mapping = WordIdxMap(word_map)

    with closing(dataset):
        try:
            _main(dataset, mapping, len(word_map) + 4)  # <start>, <unk>, <pad>, <end>
        except:  # noqa
            if get_yn_response("Remove experiment folder? [y/N]"):
                import shutil

                shutil.rmtree(experiment_folder, ignore_errors=True)
            raise


if __name__ == "__main__":
    main()
