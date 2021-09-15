import os
import glob
import random
import csv
from typing import Mapping
import mirdata
import librosa
from datetime import datetime
import sox
import soundfile as sf
import attr
import numpy as np
import torch
import yaml
import pprint
from tqdm import tqdm
from collections import OrderedDict


def slice_guitarset(data_home, save_dir, duration=5):
    """
    Slice and save guitarset audios.

    Parameters
    ----------
    data_home: str
        path to guitarset dir.
    save_dir: str
        dir to save the sliced audio files.
    duration: float
        The duration in seconds of sliced samples.
        Default is 5 seconds.
        If duration is None, return the original guitarset audios.
    
    Usage:
    import mirdata
    data_home = '/home/jovyan/workspace/datasets/guitarset'
    slice_guitarset('/your/path/to/guitarset', '/your/path/to/output', duration=5)

    """
    guitarset = mirdata.initialize('guitarset', data_home=data_home)
    tracks = guitarset.load_tracks()
    # sliced_audios = []
    # sliced_labels = []
    file_count = 0

    if duration is None: # return unsliced guitarset
        folder_name = "guitarset_nocut_clean"
        if os.path.exists(os.path.join(save_dir, folder_name)):
            raise FileExistsError(f"Output dataset already exists: {os.path.join(save_dir, folder_name)}.")
        os.mkdir(os.path.join(save_dir, folder_name))

        for track_id in tqdm(guitarset.track_ids):
            audio, sr = tracks[track_id].audio_mic # default sr of guitarset is 44100
            if sr != 44100:
                raise ValueError(f"Sample rate is not 44100 for {track_id}.")
            sf.write(
                os.path.join(save_dir, folder_name, f'{track_id}.wav'),
                audio,
                sr
                )
            file_count += 1

    else:
        folder_name = f'guitarset_{duration:.1f}s_clean'
        if os.path.exists(os.path.join(save_dir, folder_name)):
            raise FileExistsError(f"Output dataset already exists: {folder_name}.")
        os.mkdir(os.path.join(save_dir, folder_name))
        for track_id in tqdm(guitarset.track_ids):
            audio, sr = tracks[track_id].audio_mic # default sr of guitarset is 44100
            if sr != 44100:
                raise ValueError(f"Sample rate is not 44100 for {track_id}.")
            slices = librosa.util.index_to_slice(np.arange(0, len(audio), duration*sr), idx_max=len(audio))
            for i, sli in enumerate(slices):
                if len(audio[sli]) < duration*sr:
                    continue
                sf.write(
                    os.path.join(save_dir, folder_name, f"{track_id}_{i}.wav"),
                    audio[sli],
                    sr
                )
                file_count += 1
    
    print(f"Generation complete! {file_count} audio files is generated to {os.path.join(save_dir, folder_name)}.")


def gen_singleFX_1on1(clean_dirs: str, 
                      output_dir: str, 
                      fx_params:dict = None, 
                      normalize=False, 
                      random_seed=42, 
                      duration=5, 
                      add_bypass_class=True):
    """
    Generates Single FX data, each sample is used once, effect is randomly selected.

    Dataset components:
    1. generated audio files
    2. settings.yml
    3. labels_tensor.pt
        labels are -1, 0, 1, ...
        -1 is clean, others depend on fx_params
    4. clean_link.csv
        the path to corresponding clean audio for each sample
        e.g. 'dataset/clean/guitarset10/00_BN1-129-Eb_comp_0.wav'

    Parameters
    ----------
    clean_dirs: list of dirs, the audio sources
    output_dir: dir of output samples, subfolder will be created.
    fx_params: dict of fx types and parameters. if not set, use the defaul fx list.
    normalize: whether to normalize the audio
    random_seed: the random seed
    duration: duration of audio in seconds
    add_bypass_class: whether to add a bypass class. If so, bypass sample will be labeled as -1.
    """
    assert type(clean_dirs)==list

    # currently supported FX types
    LIST_SUPPORT_SFX = ['distortion',
                        'overdrive',
                        'feedback Delay',
                        'slapback Delay',
                        'reverb',
                        'chorus',
                        'flanger',
                        'phaser',
                        'tremolo',
                        # 'vibrato',
                        'low_boost',
                        'low_reduct',
                        # 'mid_boost',
                        # 'mid_reduct',
                        'hi_boost',
                        'hi_reduct',
                        ]

    print("=> Generating FX dataset...")
    # defalut fx params
    if fx_params is None:
        print("=> Using default fx params")
        fx_params = {
            'overdrive': {'gain_db': 5},
            # 'distortion': {'gain_db': 15},
            'reverb': {'reverberance': 80},
            'feedback_delay': {'n_echos': 3, 'delays': [200,400,600], 'decays':[0.4,0.2,0.1], 'gain_out':0.5},
            # 'slapback_delay': {'n_echos': 3, 'delays': [200,400,600], 'decays':[0.4,0.2,0.1], 'gain_out':0.5},
            'chorus': {'n_voices': 5},
            # 'flanger': {'depth': 5, 'phase': 50},
            # 'phaser': {},
            # 'tremolo': {},
            'low_boost': {'frequency': 200, 'gain_db': 10},
            # 'low_reduct': {'frequency': 200, 'gain_db': -10},
            'hi_boost': {'frequency': 8000, 'gain_db': 20},
            # 'hi_reduct': {'frequency': 8000, 'gain_db': -20},
        }
    else:
        print("=> Using Given fx params")
        for fx in fx_params:
            if fx not in LIST_SUPPORT_SFX:
                raise ValueError(f"Invalid or not supported effect: {fx}.")

    settings = {
        'FX_chain_type': 'single',
        'generation_type': '1on1',
        'origins': clean_dirs,
        'size': 0,
        'fx_params': OrderedDict(fx_params),
        'generate_date': datetime.now().strftime("%b-%d-%Y %H:%M:%S"),
        'nomalized': normalize,
        'sample_rate': 44100,
        'random_seed': random_seed,
        'n_classes': len(fx_params),
        'add_bypass_class': add_bypass_class
    }

    # generation start
    output_full_path = os.path.join(output_dir, f"gen_singleFX_1on1_{datetime.now().strftime('%m%d%Y')}")
    os.mkdir(output_full_path)
    print("=> Settings:")
    pprint.pprint(settings)
    sample_count = 0
    labels = [] # -1 is clean, others according to fx list
    clean_link = []
    if random_seed:
        random.seed(random_seed)

    print("=> Initializing sox transformers")
    transformers = []
    #TODO choose better sox functions for fx
    for fx in fx_params:
        transformers.append(sox.transform.Transformer())
        if fx == 'distortion':
            transformers[-1].overdrive(**fx_params[fx]) # params: {'param1': val1, 'param2', val2}
        elif fx == 'overdrive':
            transformers[-1].overdrive(**fx_params[fx])
        elif fx == 'feedback_delay':
            transformers[-1].echos(**fx_params[fx])
        elif fx == 'slapback_delay':
            transformers[-1].echo(**fx_params[fx])
        elif fx == 'reverb':
            transformers[-1].reverb(**fx_params[fx])
        elif fx == 'chorus':
            transformers[-1].chorus(**fx_params[fx])
        elif fx == 'flanger':
            transformers[-1].flanger(**fx_params[fx])
        elif fx == 'phaser':
            transformers[-1].phaser(**fx_params[fx])
        elif fx == 'tremolo':
            transformers[-1].tremolo(**fx_params[fx])
        elif fx == 'vibrato':
            raise NotImplementedError(fx)
        elif fx == 'low_boost':
            transformers[-1].bass(**fx_params[fx])
        elif fx == 'low_reduct':
            transformers[-1].bass(**fx_params[fx])
        elif fx == 'mid_boost':
            transformers[-1].equalizer(**fx_params[fx])
        elif fx == 'mid_reduct':
            transformers[-1].equalizer(**fx_params[fx])
        elif fx == 'hi_boost':
            transformers[-1].treble(**fx_params[fx])
        elif fx == 'hi_reduct':
            transformers[-1].treble(**fx_params[fx])
        else:
            raise ValueError(f"Invalid effect name: {fx}")
        transformers[-1].trim(start_time=0, end_time=duration)

    print("=> Rendering audio files")

    audio_dir = os.path.join(output_full_path, 'audio/')
    os.mkdir(audio_dir)
    for clean_dir in clean_dirs:
        settings['size'] += len(os.listdir(clean_dir))
        for sample in tqdm(os.listdir(clean_dir)):
            clean_sample_path = os.path.join(clean_dir, sample)
            clean_link.append(clean_sample_path)
            sample_audio, sr = sf.read(clean_sample_path)
            if sr != 44100:
                raise ValueError(f"Invalid sample rate: {sr}. Dataset sample rate: {settings['sample_rate']}.")
            if normalize:
                sample_audio = librosa.util.normalize(sample_audio)
            # randomly apply an fx to the sample
            if add_bypass_class:
                i = random.randint(-1, len(fx_params)-1)
            else:
                i = random.randint(0, len(fx_params)-1)
            labels.append(i)
            output_file_name = os.path.join(audio_dir, f"{sample_count}.wav")
            if i == -1:
                sf.write(output_file_name, sample_audio, settings['sample_rate'])
            else:
                transformers[i].build_file(input_array=sample_audio, sample_rate_in=settings['sample_rate'],
                                           output_filepath=output_file_name)

            sample_count += 1

    print(f"=> Generated {sample_count} samples to {output_full_path}")
    label_tensor = torch.tensor(labels)
    torch.save(label_tensor, os.path.join(output_full_path, "label_tensor.pt"))
    print(f"=> Labels written to {os.path.join(output_full_path, 'label_tensor.pt')}")
    with open(os.path.join(output_full_path, 'settings.yml'), 'w') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False)
    print(f"=> settings written to {os.path.join(output_full_path, 'settings.yml')}")
    with open(os.path.join(output_full_path, 'clean_link.csv'), 'w') as outfile:
        for sample in clean_link:
            outfile.write(sample)
            outfile.write('\n')
        # writer = csv.writer(outfile)
        # writer.writerows(clean_link)
    print(f"=> link file written to {os.path.join(output_full_path, 'clean_link.csv')}")

    print("=> Done!")




if __name__ == "__main__":
    # data_home = "/home/jovyan/workspace/datasets/guitarset"
    # slice_guitarset(data_home=data_home, save_dir="./dataset", duration=5)
    gen_singleFX_1on1(["dataset/clean/guitarset10"], "dataset/generated")
