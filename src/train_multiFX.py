from argparse import ArgumentParser
from logging import Logger
from warnings import simplefilter

import os
import pathlib
import torch
import pytorch_lightning as pl
import torchaudio
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataloader import SingleFXDataset, MultiFXDataset
from model.pl_wrapper import LightningWrapper
from model.resnet import resnet18, resnet14, resnet10, resnet6
from model.crnn import CRNN
from model.baseline import BaselineMLP
from model.sample_cnn import SampleCNN, SampleCNNSE
from transforms import MelSpectrogramDBTransform, MFCCFlatTransform

SUPPORTED_MODELS = {
    "baseline": BaselineMLP,
    "resnet18": resnet18,
    "resnet14": resnet14,
    "resnet10": resnet10,
    "resnet6": resnet6,
    "CRNN": CRNN,
    "sampleCNN": SampleCNN,
    "sampleCNNSE": SampleCNNSE,
}


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "model", type=str, help=f"model type, supported:{SUPPORTED_MODELS}"
    )
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument("--with_clean", type=bool, default=False)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("--log_class_loss", type=bool, default=False)
    parser.add_argument("--random_seed", type=int)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(42, workers=True)

    train_set = MultiFXDataset(args.data, "train", None)
    valid_set = MultiFXDataset(args.data, "valid", None)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
    )
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=8)

    print("=> Start training")
    if args.with_clean:
        print("=> Training with clean")
        in_channels = 2
    else:
        print("=> Training without clean")
        in_channels = 1

    if args.model == "baseline":
        transform = MFCCFlatTransform(n_fft=4096).cuda()
        model = BaselineMLP(2160 * in_channels, train_set.settings["n_classes"])
    elif args.model == "sampleCNN" or args.model == "sampleCNNSE":
        transform = torchaudio.transforms.Resample(44100, 22050)
        model = SUPPORTED_MODELS[args.model](
            in_channels, train_set.settings["n_classes"]
        )
    else:
        transform = MelSpectrogramDBTransform().cuda()
        model = SUPPORTED_MODELS[args.model](
            in_channels, train_set.settings["n_classes"]
        )
    wrapper = LightningWrapper(
        model, args.learning_rate, transform, args.with_clean, args.log_class_loss
    )

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5)
    log_name = f"multiFX_{model.__class__.__name__}_{str(train_set.settings['generation_type']).replace(' ','')}"
    if args.with_clean:
        log_name += "_with_clean"
    else:
        log_name += "_no_clean"
    ## log_name example: 'multiFX_resnet18_[1,5]_no_clean'
    logger = pl.loggers.TensorBoardLogger(
        pathlib.Path(__file__).parent.parent.joinpath("lightning_logs"), name=log_name
    )
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[early_stopping_callback], logger=logger, benchmark=True
    )
    trainer.fit(wrapper, train_loader, valid_loader)

    # result = trainer.test(dataloaders=val_loader)
    # print(result)


if __name__ == "__main__":
    main()
