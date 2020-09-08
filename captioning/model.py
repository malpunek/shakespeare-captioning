import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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
    """Neural Network for transforming extracted image features to semantic
    terms."""

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
        """Forward propagation. Straight after CNN extractor. Uses teacher
        forcing.

        Args:
            encoder_out (Tensor): encoded images, a tensor of dimension (batch_size,
                encoder_dim)
            encoded_captions (Tensor): encoqded captions, a tensor of dimension
                (batch_size, max_caption_length) [with <start> token and <end> token]
            caption_lengths (Tensor): length of encoded captions without <start>, <end>
                of dimension (batch_size, )
        Returns:
            tuple(Tensor, Tensor) scores for vocabulary
        """
        hidden = self.init_gru_hidden(encoder_out)  # (batch_size, decoder_dim)
        hidden = hidden.unsqueeze(
            0
        )  # GRU expects first dimension to be num_layers * num_directions
        return self.forward_hidden(hidden, encoded_captions, caption_lengths)

    def forward_hidden(self, hidden, encoded_captions, caption_lengths):
        """Forward propagation with initiated hidden state."""
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
        """Only for evaluation."""
        feats = self.extractor(img)
        terms, confidence = self.term_decoder.forward_eval(feats, mapping)
        return terms, confidence


class TermEncoder(nn.Module):
    """Encoder part of Language Generator."""

    def __init__(self, vocab_size, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        assert hidden_dim % 2 == 0

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.emb_drop = nn.Dropout(0.2)

        self.gru = nn.GRU(hidden_dim, hidden_dim // 2, bidirectional=True)
        self.gru_out_drop = nn.Dropout(0.2)
        self.gru_hid_drop = nn.Dropout(0.3)

        hidden_init = torch.zeros(2, 1, self.hidden_dim // 2)
        nn.init.normal_(hidden_init, mean=0, std=0.05)
        self.hidden_init_p = torch.nn.Parameter(hidden_init)

    def forward(self, encoded_terms, hidden, lengths):
        target_len = encoded_terms.size(1)

        embdeddings = self.embedding(encoded_terms)
        embdeddings = self.emb_drop(embdeddings)
        embdeddings = F.relu(embdeddings)

        embdeddings = torch.nn.utils.rnn.pack_padded_sequence(
            embdeddings, lengths, batch_first=True, enforce_sorted=False
        )
        out, hidden = self.gru(embdeddings, hidden)
        out, lens = torch.nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=target_len
        )
        out = self.gru_out_drop(out)  # (seq_len, batch, num_directions * hidden_size)
        hidden = self.gru_hid_drop(hidden)  # (num_directions, batch, hidden_size)

        return out, hidden, lens

    def init_hidden(self, batch_size):
        return self.hidden_init_p.expand(
            2, batch_size, self.hidden_dim // 2
        ).contiguous()


class SentenceDecoderWithAttention(nn.Module):
    """Decoder part of language generator."""

    def __init__(self, vocab_size, hidden_size, output_size, out_bias=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.emb_drop = nn.Dropout(0.2)

        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru_drop = nn.Dropout(0.2)

        self.mlp = nn.Linear(hidden_size * 2, output_size)
        if out_bias is not None:
            out_bias_tensor = torch.tensor(out_bias, requires_grad=False)
            self.mlp.bias.data[:] = out_bias_tensor
        self.logsoftmax = nn.LogSoftmax(dim=2)

        self.att_mlp = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_softmax = nn.Softmax(dim=2)

    def forward(self, input, hidden, encoder_outs, input_lengths=None):
        """Decoding

        Args:
            input (Tensor): of shape (batch, max_seq_len)
            hidden (Tensor): of shape (batch, hidden)
            encoder_outs (Tensor): of shape (batch, seq, hidden)

        Returns:
            out: all outputs (batch, output)
            hidden: last hidden state
            attn: the attention values
        """

        target_len = input.size(1)

        embeddings = self.embedding(input)  # (batch, seq_len, hidden_dim)
        embeddings = self.emb_drop(embeddings)  # (batch, seq_len, hidden_dim)
        embeddings = F.relu(embeddings)  # (batch, seq_len, hidden_dim)
        embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, input_lengths, batch_first=True, enforce_sorted=False
        )  # (batch, seq_len, hidden_dim)

        out, hidden = self.gru(embeddings, hidden)  # ()

        out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=target_len
        )  # (batch, seq_len, hidden_dim)

        out_proj = self.att_mlp(out)  # (batch, seq, hidden)
        enc_out_perm = encoder_outs.permute(0, 2, 1)  # (batch, hidden, seq)
        e_exp = torch.bmm(out_proj, enc_out_perm)
        attn = self.attn_softmax(e_exp)

        ctx = torch.bmm(attn, encoder_outs)

        full_ctx = torch.cat([self.gru_drop(out), ctx], dim=2)

        out = self.mlp(full_ctx)
        out = self.logsoftmax(out)
        return out, hidden, attn

    def forward_eval(self, encoder_out, encoder_hidden, mapping, max_len=60):
        """Forward in eval mode (without teacher forcing)

        Args:
            encoder_out (Tensor): of size (batch, seq, num_directions * hidden_size)
            encoder_hidden (Tensor): of size (num_directions, batch, hidden_size)
            mapping (dict): mapping from idxes to words
            max_len (int, optional): The maximum length of the generated caption.
                Defaults to 60.
        """

        batch_size = encoder_out.size(0)
        assert batch_size == 1

        start = mapping["<start>"]

        words_decoded = [start]
        confidence = [1]

        device = list(self.parameters())[0].device

        last_word_decoded = torch.tensor([start]).reshape((1, 1)).to(device=device)
        cap_len = torch.tensor([1]).to(device=device, dtype=torch.long)

        for _ in range(1, max_len + 1):
            out_dec, encoder_hidden, attn = self(
                last_word_decoded, encoder_hidden, encoder_out, input_lengths=cap_len
            )

            topv, topi = out_dec.topk(1)  # (batch_size, 1, 1)
            topv, topi = out_dec.topk(1)  # (batch_size, 1, 1)
            words_decoded.append(topi.item())
            confidence.append(torch.exp(topv).item())
            if topi.item() == mapping["<end>"]:
                break
            last_word_decoded = topi.reshape((1, 1)).detach()

        return list(mapping.decode(words_decoded)), confidence


class LanguageGenerator(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, terms, terms_lengths, encoded_captions, encoded_lengths):
        out, hidden, lens = self.enc(
            terms, self.enc.init_hidden(terms.size(0)), terms_lengths
        )
        hidden = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=1).unsqueeze(0)
        return self.dec(encoded_captions, hidden, out, encoded_lengths)

    def forward_eval(self, terms, terms_lengths, mapping):
        out, hidden, out_len = self.enc(
            terms, self.enc.init_hidden(terms.size(0)), terms_lengths
        )
        hidden = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=1).unsqueeze(0)
        out = out[:, : out_len.item(), :]
        return self.dec.forward_eval(out, hidden, mapping)


# TODO avoid duplication
def extract_caption_len(captions):
    captions_lens = captions[:, -1]
    captions = captions[:, :-1]
    return captions, captions_lens


class SemStyle(nn.Module):
    def __init__(self, img_to_term, language_generator, mmap, tmap, cmap):
        super().__init__()
        self.img_to_term = img_to_term
        self.language_generator = language_generator
        self.mmap = mmap
        self.tmap = tmap
        self.cmap = cmap

    def forward(self, img, style=False):
        terms, _ = self.img_to_term(img, self.mmap)
        terms = terms[1:-1]
        if not terms:
            return (terms, [], [])
        if style:
            terms = terms + ["<shake_orig>"]
        orig_terms = list(terms)
        terms = self.tmap.prepare_for_training(terms, max_caption_len=20, terms=True)
        terms = torch.LongTensor(terms).unsqueeze(0)
        terms, tlens = extract_caption_len(terms)
        return (
            orig_terms,
            *self.language_generator.forward_eval(terms, tlens, self.cmap),
        )
