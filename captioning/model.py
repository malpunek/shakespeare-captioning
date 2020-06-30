import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet101


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = nn.Sequential()
        self.resnet.out_features = self.out_features = 2048

    def forward(self, x):
        return self.resnet(x)


class TermDecoder(nn.Module):
    """
    Neural Network for transforming extracted image features to semantic terms.
    """

    def __init__(self, vocabulary_size, hidden_dim, encoder_dim, out_bias=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocabulary_size = vocabulary_size

        self.embedding = nn.Embedding(vocabulary_size, hidden_dim)
        # TODO get dropout rate from config
        self.emb_drop = nn.Dropout(0.5)

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, dropout=0.2)
        self.init_gru_hidden = nn.Linear(encoder_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, encoder_out, encoded_captions):
        """
        Forward propagation. Straight after CNN extractor.
        Uses teacher forcing.

        Args:
            encoder_out (Tensor): encoded images, a tensor of dimension (batch_size, encoder_dim)
            encoded_captions (Tensor): encoded captions, a tensor of dimension (batch_size, max_caption_length)
        Returns:
            tuple(Tensor, Tensor) scores for vocabulary
        """

        hidden = self.init_gru_hidden(encoder_out)  # (batch_size, decoder_dim)
        return self.forward_hidden(hidden, encoded_captions)

    def forward_hidden(self, hidden, encoded_captions):
        """Forward propagation with initiated hidden state"""
        embeddings = self.embedding(
            encoded_captions
        )  # (batch_size, max_caption_length, embed_dim)

        out_terms, hidden_last = self.gru(embeddings, hidden)
        out_terms = self.fc(out_terms)
        out_terms = F.log_softmax(out_terms, dim=2)

        return out_terms, hidden_last

    def forward_eval(self, encoder_out, mapping, max_len=20):
        """Forward propagation without teacher forcing.
        Currently supports only batch_size = 1

        Inspired by:
            https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#evaluation
        """

        encoder_out = self.init_gru_hidden(encoder_out)
        batch_size = encoder_out.size()[0]
        # TODO support batches
        assert batch_size == 1

        words_decoded = ["<start>"]
        assurance = [1]

        last_word_decoded = torch.tensor([mapping["start"]]).unsqueeze(0)
        # last_word_decoded = (
        #     torch.tensor(mapping["<start>"]).expand(batch_size) \
        # .reshape(batch_size, 1, 1)
        # ) # batch support

        for _ in range(1, max_len + 1):

            out_term, encoder_out = self.forward_hidden(
                encoder_out, last_word_decoded
            )  # (batch_size, 1, vocab_size)

            topv, topi = out_term.topk(1)  # (batch_size, 1, 1)
            words_decoded.append(mapping[topi.item()])
            assurance.append(topv.item())
            if topi.item() == mapping["<end>"]:
                break
            last_word_decoded = topi.unsqueeze(0).detach()
