import os
import soundfile as sf

import torch
import torchaudio
import yaml

class SingleFXDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):
        super().__init__()
        self.transform = transform
        self.audio_dir = os.path.join(data_dir, 'audio/')
        label_path = os.path.join(data_dir, 'label_tensor.pt')

        with open(os.path.join(data_dir, 'settings.yml'), 'r') as s:
            self.settings = yaml.unsafe_load(s)

        self.num_samples = len(os.listdir(self.audio_dir))
        if self.num_samples == 0:
            raise ValueError(f"No audio file found in {self.audio_dir}")

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Unable to find the label tensor in {data_dir}")
        else:
            self.labels = torch.load(label_path)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        audio, sr = torchaudio.load(os.path.join(self.audio_dir, f"{idx}.wav"))

        return self.transform(audio.view(-1)), self.labels[idx]+1
