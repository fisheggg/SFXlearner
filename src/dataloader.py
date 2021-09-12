import os
import soundfile as sf

import torch
import torchaudio

class SingleFXDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.transform = transform
        self.audio_dir = os.path.join(root, 'audio/')
        label_path = os.path.join(root, 'label_tensor.pt')

        self.num_samples = len(os.listdir(self.audio_dir))
        if self.num_samples == 0:
            raise ValueError(f"No audio file found in {self.audio_dir}")

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Unable to find the label tensor in {root}")
        else:
            self.labels = torch.load(label_path)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        audio, sr = sf.read(os.path.join(self.audio_dir, f"{idx}.wav"))

        return self.transform(audio), self.labels[idx]
