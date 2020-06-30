import torch
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

    def __init__(self, vocabulary_size, hidden_dim, out_bias=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocabulary_size = vocabulary_size

        self.embedding = nn.Embedding(vocabulary_size, hidden_dim)
        # TODO get dropout rate from config
        self.emb_drop = nn.Dropout(0.5)
        # TODO use nn.GRUs keyword dropout
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru_drop = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, vocabulary_size)

        # TODO refactor the if
        if out_bias is not None:
            out_bias_tensor = torch.tensor(out_bias, requires_grad=False)
            self.mlp.bias.data[:] = out_bias_tensor
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden_in):
        """Term decoding

        Args:
            input (Tensor): batch of sets[term idxs]. For train get them from GT
                for test start with empty and iteratively add the best token
                from previous iteration
            hidden_in (Tensor): batch of image features / output of previous
                iterations.

        Returns:
            tuple(Tensor, Tensor): (out, hidden) where out is the probability
                for each of the terms in the vocabulary; hidden is the last
                hidden state.
        """
        emb = self.embedding(input)
        emb = self.emb_drop(emb)
        out, hidden = self.gru(emb, hidden_in)
        # TODO Relu after GRU sounds good check links:
        # https://github.com/gabrielloye/GRU_Prediction/blob/master/main.ipynb
        # https://blog.floydhub.com/gru-with-pytorch/
        out = self.gru_drop(out)
        out = self.fc(out)
        out = self.logsoftmax(out)
        return out, hidden
