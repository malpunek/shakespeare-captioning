# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import (
    device,
    experiment_folder,
    second_stage,
    second_stage_dataset,
)
from ..model import LanguageGenerator, SentenceDecoderWithAttention, TermEncoder
from .misc import extract_caption_len

# In case of "RuntimeError: received 0 items of ancdata"
# https://github.com/pytorch/pytorch/issues/973
# torch.multiprocessing.set_sharing_strategy("file_system")


def train(model, dataset, mapping, criterion, optimizer, writer, epoch):

    dataloader = DataLoader(
        dataset, batch_size=second_stage["batch_size"], num_workers=4, shuffle=True
    )

    model = model.train().to(device)
    running_loss = 0
    for i, data in enumerate(tqdm(dataloader, desc="Batches")):

        caps, terms = data
        caps, terms = torch.stack(caps).to(device), torch.stack(terms).to(device)
        caps, clens = extract_caption_len(caps.T)
        terms, tlens = extract_caption_len(terms.T)

        targets = caps.detach().clone()[:, 1:]

        optimizer.zero_grad()

        out, hidden, attn = model(terms, tlens, caps[:, :-1], clens + 1)  # add <start>
        loss = criterion(out.permute(0, 2, 1), targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 50 == 49:
            step_number = epoch * len(dataloader) + i
            writer.add_scalar("Training loss", running_loss / 50, step_number)
            running_loss = 0

    return model


def main():
    dataset = second_stage_dataset()

    writer = SummaryWriter(experiment_folder)

    cmapping, tmapping = dataset.get_cap_mapping, dataset.get_term_mapping

    enc = TermEncoder(len(tmapping), 2048)
    dec = SentenceDecoderWithAttention(len(cmapping), 2048, len(cmapping))
    lang = LanguageGenerator(enc, dec)

    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = torch.optim.Adam(lang.parameters(), lr=second_stage["learning_rate"])

    for i in range(second_stage["epochs"]):
        print(f"Epoch {i}")
        lang = train(lang, dataset, cmapping, criterion, optimizer, writer, i)
        torch.save(lang.state_dict(), experiment_folder / f"language_ep{i:03d}.pth")


if __name__ == "__main__":
    main()

# %%
