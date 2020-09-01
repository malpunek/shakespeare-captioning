# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import device, experiment_folder, language_data_path, second_stage
from ..dataset import BalancedLanguageDataset
from ..model import LanguageGenerator, SentenceDecoderWithAttention, TermEncoder
from .first_stage import extract_caption_len


# https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy("file_system")


def filter_shake(shake_captions):
    return list(filter(lambda x: len(x[-1]) > 2, shake_captions))


def filter_coco(coco_caps):
    return list(filter(lambda x: len(x[-1]) > 3, coco_caps))


def train(model, dataset, mapping, criterion, optimizer, writer, epoch):

    dataloader = DataLoader(
        dataset, batch_size=second_stage["batch_size"], num_workers=8, shuffle=True
    )

    model = model.train().to(device)
    running_loss = 0
    for i, data in enumerate(tqdm(dataloader, desc="Batches")):

        caps, terms = data
        # caps, terms = next(iter(dataloader))
        caps, terms = torch.stack(caps).to(device), torch.stack(terms).to(device)
        caps, clens = extract_caption_len(caps.T)
        terms, tlens = extract_caption_len(terms.T)

        targets = caps.detach().clone()

        optimizer.zero_grad()

        out, hidden, attn = model(terms, tlens, caps.detach().clone(), clens + 1)
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
    dataset = BalancedLanguageDataset(
        language_data_path,
        filter_shakespear=filter_shake,
        filter_coco=filter_coco,
        to_tensor=True,
    )

    writer = SummaryWriter(experiment_folder)

    cmapping, tmapping = dataset.get_mappings

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
