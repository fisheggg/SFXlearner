import os
import soundfile as sf

import torch
import torchaudio
import yaml
import csv

class SingleFXDataset(torch.utils.data.Dataset):
    """
    output sample shape: [2, t]
    sample[0] is clean audio
    sample[1] is wet audio
    t is the length in time
    """
    def __init__(self, data_dir, transform):
        super().__init__()
        self.transform = transform
        self.audio_dir = os.path.join(data_dir, 'audio/')
        label_path = os.path.join(data_dir, 'label_tensor.pt')
        link_path = os.path.join(data_dir, 'clean_link.csv')

        with open(os.path.join(data_dir, 'settings.yml'), 'r') as s:
            self.settings = yaml.unsafe_load(s)

        self.num_samples = len(os.listdir(self.audio_dir))
        if self.num_samples == 0:
            raise ValueError(f"No audio file found in {self.audio_dir}")

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Unable to find the label tensor in {data_dir}")
        else:
            self.labels = torch.load(label_path)

        if not os.path.exists(link_path):
            raise FileNotFoundError(f"Unable to find the clean link in {data_dir}")
        else:
            with (open(self.clean_link), 'r') as file:
                reader = csv.reader(file)
                self.clean_link = list(reader) # 2d list, i-th string is self.clean_link[i][0]


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        audio_wet, _ = torchaudio.load(os.path.join(self.audio_dir, f"{idx}.wav"))
        audio_clean, _ = torchaudio.load(self.clean_link[idx][0])
        

        if self.settings['add_bypass_class']:
            return self.transform(torch.cat(audio_clean, audio_wet)), self.labels[idx]+1
        else:
            return self.transform(torch.cat(audio_clean, audio_wet)), self.labels[idx]
