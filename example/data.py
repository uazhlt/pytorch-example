import torch
from torchtext.data import TabularDataset, Field, LabelField
import torchtext



class DataUtils(object):

    # our tokenizer.  
    # Here we're tokenizing to characters.
    char_tokenizer = lambda x: list(x)

    @staticmethod
    def ngrams(seq, n=1):
        grams = []
        seq = ["^"] + seq + ["$"]
        n = min(len(seq)-1, n)
        for i in range(len(seq) - n + 1):
            grams.append("".join(seq[i:i+n]))
        return grams or "".join(seq)
    
    # transformations (if any) to apply after tokenization.
    # For example, we might want to use bigrams 
    # and prepend and append special characters to indicate the start and end of the sequence.
    unigrams   = lambda x: DataUtils.ngrams(x, n=1)
    bigrams    = lambda x: DataUtils.ngrams(x, n=2)

    PAD_SYM    = "<pad>"
    # define field processors

    @staticmethod
    def input_field(tokenizer, preprocessor, lowercase=True):
        return Field(
            sequential=True, 
            tokenize=tokenizer,
            preprocessing=preprocessor,
            pad_token=DataUtils.PAD_SYM,
            lower=lowercase,
            batch_first=True # currently crucial for getting right order
        )

    @staticmethod
    def label_field(): return LabelField(sequential=False)

    @staticmethod
    def load_datasets(data_dir, input_field, label_field):
        # dataset derived from https://github.com/joosthub/PyTorchNLPBook/tree/master/data#surnames
        # train_ds, dev_ds, test_ds = 
        return TabularDataset.splits(
            path=data_dir, 
            train="surnames-train.csv",
            validation="surnames-dev.csv", 
            test="surnames-test.csv", 
            format='csv',
            skip_header=False,
            fields={ 
                "surname": ("name", input_field),
                "nationality": ("label", label_field)
            }
        )

    @staticmethod
    def train(model, device, training_data, optimizer, criterion):
        """
        Method to train a model on the provided data.

        :param model: PyTorch model to train.
        :type model: subclass of nn.Module

        :param device: specifies whether to run on a CPU or GPU. See https://pytorch.org/docs/stable/tensor_attributes.html#torch-device
        :type device: torch.device

        :param training_data: batched dataset for training.
        :type training_data: torchtext.data.iterator.BucketIterator

        :param optimizer: PyTorch optimizer (ex. SGD, ADAM, etc.; see https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)
        :type optimizer: torch.optim.Optimizer

        :param criterion: PyTorch loss function (ex. CrossEntropy, etc.; see https://pytorch.org/docs/stable/nn.html#loss-functions)
        """
        # ensure model is in training mode
        model.train()
        # Train the model
        train_loss = 0
        for batch in training_data:
            optimizer.zero_grad()
            x, y_true = batch.name.to(device), batch.label.to(device)
            y_hat     = model(x, softmax=False)
            loss      = criterion(y_hat, y_true)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        return train_loss / len(training_data)

    @staticmethod
    def validation_monitor(model, device, validation_data, criterion):

        """
        Method for monitoring loss on validation data.  Used to help prevent overfitting.

        :param model: PyTorch model to use for inference.
        :type model: subclass of nn.Module

        :param device: specifies whether to run on a CPU or GPU. See https://pytorch.org/docs/stable/tensor_attributes.html#torch-device
        :type device: torch.device

        :param validation_data: batched validation dataset used to avoid overfitting.
        :type data: torchtext.data.iterator.BucketIterator

        :param criterion: PyTorch loss function (ex. CrossEntropy, etc.; see https://pytorch.org/docs/stable/nn.html#loss-functions)
        """
        loss = 0
        #acc = 0
        for batch in validation_data:
            x, y_true = batch.name.to(device), batch.label.to(device)
            with torch.no_grad():
                y_hat = model(x, softmax=False)
                loss = criterion(y_hat, y_true)
                loss += loss.item()
                #acc += (y_hat.argmax() == y_true).sum().item()

        return loss / len(validation_data)