from models import DANish
from data import DataUtils

import argparse
import torch
import torchtext
import os
import sys




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train PyTorch classifier on surnames data.')

    parser.set_defaults(func=lambda x: parser.print_usage())
    
    parser.add_argument(
        '--epochs',
        required=True,
        type=int,
        help='Number of epochs to train'
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        type=str,
        help='Parent directory of training data'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        type=str,
        help='Directory for checkpointed models'
    )

    parser.add_argument(
        '--batchsize', '-b',
        required=True,
        type=int,
        help='The maximum size of each minibatch'
    )

    args = parser.parse_args()
    
    #print(args)

    INPUT_DIR  = os.path.abspath(args.input)
    OUT_DIR    = os.path.abspath(args.output)

    BATCH_SIZE = args.batchsize
    N_EPOCHS   = args.epochs

    print()
    print(f"INPUT_DIR:\t{INPUT_DIR}")
    print(f"OUT_DIR:\t{OUT_DIR}")
    print(f"N_EPOCHS:\t{N_EPOCHS}")
    print(f"BATCH_SIZE:\t{BATCH_SIZE}")
    print()

    print(f"Will save model snapshots to {OUT_DIR}/\n")
    os.makedirs(OUT_DIR)

    # FIXME: make CLI param
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create preprocessor for input
    TEXT = DataUtils.input_field(tokenizer=DataUtils.char_tokenizer, preprocessor=DataUtils.bigrams)
    # create preprocessor for labels
    LABEL = DataUtils.label_field()

    # load data
    # FIXME: make data_dir a CLI param
    train_ds, dev_ds, test_ds = DataUtils.load_datasets(data_dir=INPUT_DIR, input_field=TEXT, label_field=LABEL)

    # tally vocabularies for input and labels
    TEXT.build_vocab(train_ds, min_freq=1)
    LABEL.build_vocab(train_ds)
    #LABEL.vocab.freqs.most_common()
    #LABEL.vocab.stoi

    model = DANish(
        vocab_size=len(TEXT.vocab), 
        embedding_dim=50, # FIXME: make CLI param
        hidden_dim=200, # FIXME: make CLI param
        num_classes=len(LABEL.vocab),
        padding_idx=TEXT.vocab.stoi[DataUtils.PAD_SYM]
    )

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    for epoch in range(1, N_EPOCHS + 1):
        # split our data into batches
        train_iter, dev_iter = torchtext.data.BucketIterator.splits(
            datasets=(train_ds, dev_ds), 
            batch_sizes=(BATCH_SIZE, BATCH_SIZE),
            repeat=False,
            shuffle=True,
            sort_key=lambda x: len(x.name), # needed for shuffling
            device=device
        )
        train_loss = DataUtils.train(
            model=model,
            device=device,
            training_data=train_iter,
            optimizer=optimizer,
            criterion=criterion
        )
        print(f"Epoch {epoch}:")
        print(f"\ttrain loss: {train_loss:.6f}")
        print()
        if epoch % 10 == 0:
            print(f"Checkpointing results for Epoch {epoch}...\n")
            out_file = os.path.join(OUT_DIR, f"danish-epoch-{epoch}")

            # Ensure .state_dict() is stored.
            # store any additional key-value pairs you find informative (ex. loss, dev performance)
            torch.save({
                'epoch': epoch,
                'train_loss': train_loss,
                'model_state_dict': model.state_dict()
            }, out_file)
    pass
