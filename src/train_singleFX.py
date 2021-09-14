from argparse import ArgumentParser
from warnings import simplefilter

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torchaudio
from dataloader import SingleFXDataset
from model.sample_model import VanillaNN
from transforms import MFCCSumTransform

def main():
    parser = ArgumentParser()
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('--n_mfcc', type=int, default=40)
    parser.add_argument('--val_split', type=float, default=0.2)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()


    transform = MFCCSumTransform(sample_rate=44100, n_mfcc=args.n_mfcc)
    dataset = SingleFXDataset(args.data, transform)
    if args.val_split > 0:
        val_size = int(dataset.settings['size'] * args.val_split)
        train_set, test_set = random_split(dataset, [dataset.settings['size']-val_size, val_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=1)
        val_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=1)
    else:
        train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)
        val_loader = None

    model = VanillaNN(input_dim=args.n_mfcc, num_classes=dataset.settings['n_classes']+1, lr=args.learning_rate)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    result = trainer.test(dataloaders=val_loader)
    print(result)

if __name__ == '__main__':
    main()