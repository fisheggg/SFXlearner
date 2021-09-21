from argparse import ArgumentParser
from logging import Logger
from warnings import simplefilter

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torchaudio
from dataloader import SingleFXDataset
from model.sample_model import VanillaNN, VanillaNNWithClean
from transforms import MFCCSumTransform

def main():
    parser = ArgumentParser()
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--with_clean', default=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('--n_mfcc', type=int, default=40)
    parser.add_argument('--val_split', type=float, default=0.2)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    transform = MFCCSumTransform(sample_rate=44100, n_mfcc=args.n_mfcc)
    train_set = SingleFXDataset(args.data, 'train', transform)
    valid_set = SingleFXDataset(args.data, 'valid', transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=16, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=16)

    print("=> Start training")
    args.with_clean = True
    if args.with_clean is True:
        print("=> Training with clean")
        model = VanillaNNWithClean(input_dim=args.n_mfcc, num_classes=train_set.settings['n_classes'], lr=args.learning_rate)
    else:
        print("=> Training no clean")
        model = VanillaNN(input_dim=args.n_mfcc, num_classes=train_set.settings['n_classes'], lr=args.learning_rate)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, valid_loader)

    # result = trainer.test(dataloaders=val_loader)
    # print(result)

if __name__ == '__main__':
    main()