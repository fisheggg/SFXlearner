from argparse import ArgumentParser
from logging import Logger
from warnings import simplefilter

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torchaudio
from dataloader import SingleFXDataset, MultiFXDataset
from model.resnet import resnet18

def main():
    parser = ArgumentParser()
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--with_clean', default=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    transform = torchaudio.transforms.MelSpectrogram(sample_rate=44100,
                                                      n_fft=2048,
                                                      n_mels=128)
    train_set = MultiFXDataset(args.data, 'train', None)
    valid_set = MultiFXDataset(args.data, 'valid', None)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=16, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=16)

    print("=> Start training")
    args.with_clean = True
    if args.with_clean is True:
        print("=> Training with clean")
        in_channels = 2
    else:
        print("=> Training no clean")
        in_channels = 1

    model = resnet18(in_channels, train_set.settings["n_classes"], args.with_clean, args.learning_rate, transform)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, valid_loader)

    # result = trainer.test(dataloaders=val_loader)
    # print(result)

if __name__ == '__main__':
    main()