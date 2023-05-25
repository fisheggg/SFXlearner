import os
import torch
import pytorch_lightning as pl
import argparse
import pandas as pd
import pprint
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report
from transforms import *
from dataloader import MultiFXDataset
from model.pl_wrapper import LightningWrapper
from model.resnet import *
from model.crnn import *
from model.baseline import *
from model.sample_cnn import SampleCNN, SampleCNNSE

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


def generate_evaluation_report(
    model_name: str,
    checkpoint_dir: str,
    test_dir: str,
    save_dir: str,
    with_clean: bool,
    verbose: bool = True,
):
    """
    Generates an evalutation report
    """
    if verbose:
        print("=> Loading checkpoint")
    if with_clean:
        print("=> Inferencing with clean")
        in_channels = 2
    else:
        print("=> Inferencing without clean")
        in_channels = 1

    if model_name == "baseline":
        transform = MFCCFlatTransform(n_fft=4096).cuda()
        model = BaselineMLP(2160 * in_channels, 13)
    elif args.model == "sampleCNN" or args.model == "sampleCNNSE":
        transform = torchaudio.transforms.Resample(44100, 22050)
        model = SUPPORTED_MODELS[args.model](in_channels, 13)
    else:
        transform = MelSpectrogramDBTransform(
            sample_rate=44100, n_fft=2048, n_mels=128
        ).cuda()
        model = SUPPORTED_MODELS[model_name](in_channels=in_channels, num_classes=13)

    checkpoint = os.listdir(os.path.join(checkpoint_dir, "checkpoints"))[0]
    checkpoint = os.path.join(checkpoint_dir, "checkpoints", checkpoint)

    wrapper = LightningWrapper.load_from_checkpoint(
        checkpoint, model=model, lr=1e-3, transform=transform
    )
    wrapper.eval()
    valid_set = MultiFXDataset(test_dir, "valid", None)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=64, num_workers=2)
    trainer = pl.Trainer(gpus=1)
    if verbose:
        print("=> Inferecing test set")
    wrapper.cuda()
    y = torch.randn(1, 13, requires_grad=False).cuda()
    y_hat = torch.randn(1, 13, requires_grad=False).cuda()
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            x, y_batch = batch
            if not with_clean:
                x = x[:, 1, :].unsqueeze(1)
            y_hat_batch = wrapper(x.cuda())
            y = torch.vstack([y, y_batch.cuda()])
            y_hat = torch.vstack([y_hat, y_hat_batch.cuda()])
            del x, y_batch, y_hat_batch
            torch.cuda.empty_cache()

    y_hat = torch.sigmoid(y_hat) > 0.5
    labels = list(valid_set.settings["fx_params"].keys())
    report = classification_report(
        y[1:, :].cpu().to(torch.int),
        y_hat[1:, :].cpu().to(torch.int),
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )
    # breakpoint()
    folder_name = f"{model_name}_{test_dir.split('/')[-2][12:]}"
    if with_clean:
        folder_name += "_with_clean"
    else:
        folder_name += "_no_clean"
    if save_dir:
        save_path = os.path.join(save_dir, folder_name)
        if verbose:
            print(f"=> saving report to {save_path}")
        os.mkdir(save_path)
        sheet = pd.DataFrame.from_dict(report, orient="index")
        sheet.to_csv(os.path.join(save_path, "report.csv"))
    if verbose:
        print()
        print("=" * 20)
        pprint.pprint(report)
        print()


if __name__ == "__main__":
    supported_models = list(SUPPORTED_MODELS.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=str, help=f"model type, supported:{supported_models}"
    )
    parser.add_argument(
        "checkpoint_dir", metavar="DIR", help="path to model checkpoint"
    )
    parser.add_argument("test_dir", metavar="DIR", help="path to test dataset")
    parser.add_argument(
        "save_dir",
        metavar="DIR",
        default=Path(__file__).parent.absolute(),
        help="path to save evaluation result",
    )
    parser.add_argument("--with_clean", type=bool, default=False)
    args = parser.parse_args()

    generate_evaluation_report(
        args.model, args.checkpoint_dir, args.test_dir, args.save_dir, args.with_clean
    )
