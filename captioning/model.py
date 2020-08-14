import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet101
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

        self.embedding = nn.Embedding(vocabulary_size, hidden_dim, padding_idx=0)
        # TODO get dropout rate from config
        self.emb_drop = nn.Dropout(0.5)

        self.gru = nn.GRU(hidden_dim, hidden_dim, dropout=0.2)
        self.init_gru_hidden = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim), nn.ReLU()
        )

        self.fc = nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation. Straight after CNN extractor.
        Uses teacher forcing.

        Args:
            encoder_out (Tensor): encoded images, a tensor of dimension (batch_size, encoder_dim)
            encoded_captions (Tensor): encoded captions, a tensor of dimension (batch_size, max_caption_length)
                [with <start> token and <end> token]
            caption_lengths (Tensor): length of encoded captions without <start>, <end> (batch_size, )
        Returns:
            tuple(Tensor, Tensor) scores for vocabulary
        """
        hidden = self.init_gru_hidden(encoder_out)  # (batch_size, decoder_dim)
        hidden = hidden.unsqueeze(
            0
        )  # GRU expects first dimension to be num_layers * num_directions
        return self.forward_hidden(hidden, encoded_captions, caption_lengths)

    def forward_hidden(self, hidden, encoded_captions, caption_lengths):
        """Forward propagation with initiated hidden state"""
        target_len = encoded_captions.size(1)
        embeddings = self.embedding(
            encoded_captions
        )  # (batch_size, max_caption_length, embed_dim)
        embeddings = self.emb_drop(embeddings)
        embeddings = F.relu(embeddings)  # inspired by pytorch nlp
        embeddings = pack_padded_sequence(
            embeddings, caption_lengths, batch_first=True, enforce_sorted=False
        )
        out_terms, hidden_last = self.gru(embeddings, hidden)
        # out_terms: (seq_len, batch, hidden_dim), hidden_last: (1, batch, hidden_dim)
        # TODO: Original paper proposes GRU_DROPOUT[=nn.Dropuot(0.5)](out_terms) here

        out_terms, out_lengths = pad_packed_sequence(
            out_terms, batch_first=True, total_length=target_len
        )
        # (batch, max_len, hidden_dim)
        out_terms = self.fc(out_terms)
        # (batch, max_len, vocab_size)
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
        assert batch_size == 1
        encoder_out = encoder_out.reshape(1, batch_size, -1)
        # TODO support batches

        start = mapping["<start>"]

        words_decoded = [start]
        confidence = [1]

        device = list(self.parameters())[0].device

        last_word_decoded = torch.tensor([start]).reshape((1, 1)).to(device=device)
        cap_len = torch.tensor([1]).to(device=device, dtype=torch.long)

        # last_word_decoded = (
        #     torch.tensor(mapping["<start>"]).expand(batch_size) \
        # .reshape(batch_size, 1, 1)
        # ) # batch support

        for _ in range(1, max_len + 1):

            out_term, encoder_out = self.forward_hidden(
                encoder_out, last_word_decoded, cap_len
            )  # (batch_size, 1, vocab_size)

            topv, topi = out_term.topk(1)  # (batch_size, 1, 1)
            words_decoded.append(topi.item())
            confidence.append(torch.exp(topv).item())
            if topi.item() == mapping["<end>"]:
                break
            last_word_decoded = topi.reshape((1, 1)).detach()

        return list(mapping.decode(words_decoded)), confidence


class ImgToTermNet(nn.Module):
    def __init__(self, term_decoder, extractor=None):
        super().__init__()
        self.term_decoder = term_decoder
        self.extractor = extractor or FeatureExtractor()

    def forward(self, img, mapping):
        """Only for evaluation"""
        feats = self.extractor(img)
        terms, confidence = self.term_decoder.forward_eval(feats, mapping)
        return terms, confidence

