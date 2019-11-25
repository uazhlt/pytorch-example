import torch.nn as nn
import torch.nn.functional as F


class DANish(nn.Module):
    """
    Sloppy/simplified version of a deep averaging network (DAN) classifier. DAN-like. DANish.
    
    See https://aclweb.org/anthology/P15-1162/

    NOTE: Word dropout should be applied as preprocessing step (i.e., before feeding data into the network).

    :param vocab_size: the size of the vocabulary.  Used to instantiate the embeddings table.
    :type vocab_size: int
    
    :param embedding_dim: the number of dimensions to use for each embedding.
    :type embedding_dim: int

    :param hidden_dim: the size of the hidden layer.
    :type hidden_dim: int

    :param hidden_dim: the size of the hidden layer.
    :type hidden_dim: int

    :param padding_idx: the index for the padding symbol.  Should be the same between the vocabulary and embedding table.
    :type padding_idx: int

    :param num_classes: the number of classes.  Used to instantiate the final layer.
    :type num_classes: int
    """
    def __init__(
        self, 
        vocab_size, 
        embedding_dim, 
        hidden_dim, 
        padding_idx, 
        num_classes
    ):
        super(DANish, self).__init__()
        self.padding_idx = padding_idx

        self.embeddings  = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            sparse=False # NOTE: few optimizers support sparse gradients
        )
        self.fc1         = nn.Linear(embedding_dim, hidden_dim)
        self.fc2         = nn.Linear(hidden_dim, num_classes)
        #self.init_weights()
    
    def init_weights(self):
        """
        Initializes weights
        """
        initrange = 0.5
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, batch_x, softmax=False):
        emb         = self.embeddings(batch_x)
        # count num. non-padding embeddings per datum
        nonpad_cnts = (batch_x != self.padding_idx).sum(dim=-1).view(-1, 1)
        # mean that ignores padding
        composition = emb.sum(dim=1) / nonpad_cnts
        # feed to first layer and apply nonlinearity (ex. ReLU)
        h           = F.relu(self.fc1(composition))
        output      = self.fc2(h)
        if softmax:
            output = F.softmax(output, dim=1)
        return output