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
from sox import transform
import torch
from torch.serialization import validate_cuda_device
import yaml
import pprint
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import OrderedDict
from itertools import combinations, product

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

def gen_singleFX_1on1(clean_dirs: list, 
                      output_dir: str, 
                      fx_params: dict = None, 
                      normalize: bool = False, 
                      random_seed: int = 42, 
                      duration: float = 5, 
                      add_bypass_class: bool = False):
    """
    Generates Single FX data, each sample is used once, effect is randomly selected.

    Dataset components:
    1. generated audio files
    2. settings.yml
    3. labels_tensor.pt
        labels are -1, 0, 1, ...
        -1 is bypass (if add_bypass_class == True), others depend on fx_params
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
            'distortion': {'gain_db': 15},
            'reverb': {'reverberance': 80},
            'feedback_delay': {'n_echos': 3, 'delays': [200,400,600], 'decays':[0.4,0.2,0.1], 'gain_out':0.5},
            'slapback_delay': {'n_echos': 3, 'delays': [200,400,600], 'decays':[0.4,0.2,0.1], 'gain_out':0.5},
            'chorus': {'n_voices': 5},
            'flanger': {'depth': 5, 'phase': 50},
            'phaser': {},
            'tremolo': {},
            'low_boost': {'frequency': 200, 'gain_db': 10},
            'low_reduct': {'frequency': 200, 'gain_db': -10},
            'hi_boost': {'frequency': 8000, 'gain_db': 20},
            'hi_reduct': {'frequency': 8000, 'gain_db': -20},
        }
    else:
        print("=> Using Given fx params")
        for fx in fx_params:
            if fx not in LIST_SUPPORT_SFX:
                raise ValueError(f"Invalid or not supported effect: {fx}.")

    settings = {
        'fx_chain_type': 'single',
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

def gen_singleFX_1onN(clean_dirs: list, 
                      output_dir: str, 
                      fx_params: dict = None, 
                      normalize: bool = False, 
                      random_seed: int = 42, 
                      duration: float = 5,
                      valid_split: float = 0.2, 
                      add_bypass_class: bool = False):
    """
    Generates Single FX data, each sample is used for all effects.
    output_size = clean_size * (n_classes (+1 if add_bypass_class))

    Dataset components:
    root/
    |---settings.yml
    |---train/
        |---audio/
            |---0.wav
            |---1.wav
            ...
        |---clean_link.csv
        |---label_tensor.pt
    |---valid/
        |---audio/
            |---0.wav
            |---1.wav
            ...
        |---clean_link.csv
        |---label_tensor.pt    

    1. generated audio files
    2. settings.yml
    3. labels_tensor.pt
        labels are -1, 0, 1, ...
        -1 is bypass (if add_bypass_class == True), others depend on fx_params
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
            'distortion': {'gain_db': 15},
            'reverb': {'reverberance': 80},
            'feedback_delay': {'n_echos': 3, 'delays': [200,400,600], 'decays':[0.4,0.2,0.1], 'gain_out':0.5},
            'slapback_delay': {'n_echos': 3, 'delays': [200,400,600], 'decays':[0.4,0.2,0.1], 'gain_out':0.5},
            'chorus': {'n_voices': 5},
            'flanger': {'depth': 5, 'phase': 50},
            'phaser': {},
            'tremolo': {},
            'low_boost': {'frequency': 200, 'gain_db': 10},
            'low_reduct': {'frequency': 200, 'gain_db': -10},
            'hi_boost': {'frequency': 8000, 'gain_db': 20},
            'hi_reduct': {'frequency': 8000, 'gain_db': -20},
        }
    else:
        print("=> Using Given fx params")
        for fx in fx_params:
            if fx not in LIST_SUPPORT_SFX:
                raise ValueError(f"Invalid or not supported effect: {fx}.")

    settings = {
        'fx_chain_type': 'single',
        'generation_type': '1onN',
        'origins': clean_dirs,
        'train_size': 0,
        'valid_size': 0,
        'fx_params': OrderedDict(fx_params),
        'generate_date': datetime.now().strftime("%b-%d-%Y %H:%M:%S"),
        'nomalized': normalize,
        'sample_rate': 44100,
        'random_seed': random_seed,
        'n_classes': len(fx_params),
        'add_bypass_class': add_bypass_class,
    }

    # generation start
    output_full_path = os.path.join(output_dir, f"gen_singleFX_1onN_{datetime.now().strftime('%m%d%Y')}")
    os.mkdir(output_full_path)
    print("=> Settings:")
    pprint.pprint(settings)
    train_sample_count = 0
    valid_sample_count = 0
    train_labels = [] # -1 is clean, others according to fx list
    valid_labels = []
    train_clean_link = []
    valid_clean_link = []
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

    train_audio_dir = os.path.join(output_full_path, 'train/audio/')
    valid_audio_dir = os.path.join(output_full_path, 'valid/audio/')
    os.makedirs(train_audio_dir)
    os.makedirs(valid_audio_dir)
    for clean_dir in clean_dirs:
        clean_paths = os.listdir(clean_dir)
        train_paths, valid_paths = train_test_split(clean_paths, 
                                                   test_size=valid_split, 
                                                   random_state=random_seed,
                                                   shuffle=True)
        if add_bypass_class:
            settings['train_size'] += len(train_paths) * (len(fx_params)+1)
            settings['valid_size'] += len(valid_paths) * (len(fx_params)+1)
        else:
            settings['train_size'] += len(train_paths) * (len(fx_params)+1)
            settings['valid_size'] += len(valid_paths) * (len(fx_params)+1)

        print(f"=> Generating training set from {clean_dir}")
        for sample in tqdm(train_paths):
            clean_sample_path = os.path.join(clean_dir, sample)
            sample_audio, sr = sf.read(clean_sample_path)
            if sr != 44100:
                raise ValueError(f"Invalid sample rate: {sr}. Dataset sample rate: {settings['sample_rate']}.")
            if normalize:
                sample_audio = librosa.util.normalize(sample_audio)
            # randomly apply an fx to the sample
            if add_bypass_class:
                i_range = range(-1, len(fx_params))
            else:
                i_range = range(0, len(fx_params))
            for i in i_range:
                train_clean_link.append(clean_sample_path)
                train_labels.append(i)
                output_file_name = os.path.join(train_audio_dir, f"{train_sample_count}.wav")
                if i == -1:
                    sf.write(output_file_name, sample_audio, settings['sample_rate'])
                else:
                    transformers[i].build_file(input_array=sample_audio, sample_rate_in=settings['sample_rate'],
                                           output_filepath=output_file_name)
                train_sample_count += 1
        print(f"=> Generating valid set from {clean_dir}")
        for sample in tqdm(valid_paths):
            clean_sample_path = os.path.join(clean_dir, sample)
            sample_audio, sr = sf.read(clean_sample_path)
            if sr != 44100:
                raise ValueError(f"Invalid sample rate: {sr}. Dataset sample rate: {settings['sample_rate']}.")
            if normalize:
                sample_audio = librosa.util.normalize(sample_audio)
            # randomly apply an fx to the sample
            if add_bypass_class:
                i_range = range(-1, len(fx_params))
            else:
                i_range = range(0, len(fx_params))
            for i in i_range:
                valid_clean_link.append(clean_sample_path)
                valid_labels.append(i)
                output_file_name = os.path.join(valid_audio_dir, f"{valid_sample_count}.wav")
                if i == -1:
                    sf.write(output_file_name, sample_audio, settings['sample_rate'])
                else:
                    transformers[i].build_file(input_array=sample_audio, sample_rate_in=settings['sample_rate'],
                                           output_filepath=output_file_name)
                valid_sample_count += 1


    print(f"=> Generated {train_sample_count} samples to {output_full_path}/train")
    print(f"=> Generated {valid_sample_count} samples to {output_full_path}/valid")

    train_label_tensor = torch.tensor(train_labels)
    valid_label_tensor = torch.tensor(valid_labels)
    torch.save(train_label_tensor, os.path.join(f"{output_full_path}/train", "label_tensor.pt"))
    print(f"=> Training set labels written to {os.path.join(output_full_path, 'train/', 'label_tensor.pt')}")
    torch.save(valid_label_tensor, os.path.join(f"{output_full_path}/valid", "label_tensor.pt"))
    print(f"=> Validation set labels written to {os.path.join(output_full_path, 'valid/', 'label_tensor.pt')}")

    with open(os.path.join(output_full_path, 'train/', 'clean_link.csv'), 'w') as outfile:
        for sample in train_clean_link:
            outfile.write(sample)
            outfile.write('\n')
    print(f"=> Training link file written to {os.path.join(output_full_path, 'train/', 'clean_link.csv')}")
    with open(os.path.join(output_full_path, 'valid/', 'clean_link.csv'), 'w') as outfile:
        for sample in valid_clean_link:
            outfile.write(sample)
            outfile.write('\n')
    print(f"=> Validation link file written to {os.path.join(output_full_path, 'valid/', 'clean_link.csv')}")

    with open(os.path.join(output_full_path, 'settings.yml'), 'w') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False)
    print(f"=> settings written to {os.path.join(output_full_path, 'settings.yml')}")
    print("=> Done!")

def gen_multiFX(clean_dirs: list,
                output_dir: str,
                methods: list,
                fx_params: dict = None,
                grouping: list = None,
                normalize: bool = False,
                random_seed: int = 42,
                duration: float = 5,
                valid_split: float = 0.2,
                add_bypass_class: bool = False):
    """
    Generates multi-FX data, each sample is used multiple times according to methods list.
    output size is decided by the methods list. See descriptions below.

    Dataset components:
    root/
    |---settings.yml
    |---train/
        |---audio/
            |---0.wav
            |---1.wav
            ...
        |---clean_link.csv
        |---label_tensor.pt
    |---valid/
        |---audio/
            |---0.wav
            |---1.wav
            ...
        |---clean_link.csv
        |---label_tensor.pt    

    1. generated audio files
    2. settings.yml
    3. labels_tensor.pt
        shape: (n_sampels, n_classes)
    4. clean_link.csv
        the path to corresponding clean audio for each sample
        e.g. 'dataset/clean/guitarset10/00_BN1-129-Eb_comp_0.wav'
    
    Parameters
    ----------
    clean_dirs: list of dirs, the audio sources
    output_dir: dir of output samples, subfolder will be created.
    methods: list of int or strings, how the effect chains are generated.
             int N will apply N groups of effects. 
                It will iterate over all combinations of groups, and all effects within a group.
             can have multiple Ns.
             N must in [1, n_groups].
             valid strings: "random"
    fx_params: dict of fx types and parameters. if not set, use the defaul fx list.
                each dict represents an FX group.
    grouping: list of tuples, indicates the grouping of effects.
              e.g. [(0, 1), (2), (3, 4)]
    normalize: whether to normalize the output audio
    random_seed: the random seed
    duration: duration of audio in seconds   
    """
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
                        'low_boost',
                        'low_reduct',
                        'hi_boost',
                        'hi_reduct',
                        ]
    print("=> Generating FX dataset...")
    # defalut fx params
    if fx_params is None:
        print("=> Using default fx params and grouping")
        fx_params = {
            'overdrive': {'gain_db': 5},
            'distortion': {'gain_db': 15},
            'reverb': {'reverberance': 80},
            'feedback_delay': {'n_echos': 3, 'delays': [200,400,600], 'decays':[0.4,0.2,0.1], 'gain_out':0.5},
            'slapback_delay': {'n_echos': 3, 'delays': [200,400,600], 'decays':[0.4,0.2,0.1], 'gain_out':0.5},
            'chorus': {'n_voices': 5},
            'flanger': {'depth': 5, 'phase': 50},
            'phaser': {},
            'tremolo': {},
            'low_boost': {'frequency': 200, 'gain_db': 10},
            'low_reduct': {'frequency': 200, 'gain_db': -10},
            'hi_boost': {'frequency': 8000, 'gain_db': 20},
            'hi_reduct': {'frequency': 8000, 'gain_db': -20},
        }
        grouping = [
            (0, 1,),
            (2,),
            (3, 4,),
            (5, 6, 7, 8),
            (9, 10,),
            (11, 12,)
        ]
    else:
        if grouping is None:
            raise ValueError(f"FX dict is provided but no grouping list.")
        print("=> Using Given fx params")
        for fx in fx_params:
            if fx not in LIST_SUPPORT_SFX:
                raise ValueError(f"Invalid or not supported effect: {fx}.")
    n_groups = len(grouping)

    settings = {
        'fx_chain_type': 'multi',
        'generation_type': methods,
        'origins': clean_dirs,
        'train_size': 0,
        'valid_size': 0,
        'fx_params': OrderedDict(fx_params),
        'grouping': grouping,
        'n_groups': n_groups,
        'generate_date': datetime.now().strftime("%b-%d-%Y %H:%M:%S"),
        'nomalized': normalize,
        'sample_rate': 44100,
        'random_seed': random_seed,
        'n_classes': len(fx_params),
        'add_bypass_class': add_bypass_class,
    }
    for method in methods:
        if type(method) == int:
            assert method >= 1 and method <= n_groups
        elif type(method) == str:
            assert method == "random"
        else:
            raise ValueError(f"Invalid method: {method}")

    # generation start
    output_full_path = os.path.join(output_dir, f"gen_multiFX_{datetime.now().strftime('%m%d%Y')}")
    os.mkdir(output_full_path)
    print("=> Settings:")
    pprint.pprint(settings)
    train_sample_count = 0
    valid_sample_count = 0
    train_labels = [] # shape: (n_samples, n_classes)
    valid_labels = []
    train_clean_link = []
    valid_clean_link = []
    if random_seed:
        random.seed(random_seed)

    print("=> Initializing sox transformers")
    transformers = []
    transformers_labels = [] #shape: (len(transformers), n_classes)
    fx_name_list = list(fx_params.keys())
    for method in methods:
        if type(method) is int:
            for groups_to_apply in combinations(range(n_groups), method):
                # for one method N, find all the combinations of groups
                # e.g N=5, then groups_to_apply=((0,1,2,3,4),(0,1,2,3,5),(0,1,2,4,5),(0,1,3,4,5),(0,2,3,4,5),(1,2,3,4,5))
                fx_chain_iter = product(*list(map(lambda x: grouping[x], groups_to_apply)))
                # for one fixed groups_to_apply, get all the combinations of fx. Iterates over all fx combinations amoung groups
                # e.g. groups_to_apply = (0,1,2), then fx_chain_iter=((0,2,3),(0,2,4),(1,2,3),(1,2,4))
                for fx_chain in fx_chain_iter:
                    transformers.append(sox.transform.Transformer())
                    label = [0]*len(fx_params)
                    for i in fx_chain:
                        label[i] = 1
                    transformers_labels.append(label)
                    for i in fx_chain:
                        fx = fx_name_list[i]
                        apply_fx_to_transformer(transformers[-1], fx, fx_params)
                        transformers[-1].trim(start_time=0, end_time=duration)
                        transformers[-1].set_output_format(bits=16)
        elif method == "random":
            #TODO
            pass

    print("=> Rendering audio files")

    train_audio_dir = os.path.join(output_full_path, 'train/audio/')
    valid_audio_dir = os.path.join(output_full_path, 'valid/audio/')
    os.makedirs(train_audio_dir)
    os.makedirs(valid_audio_dir)
    for clean_dir in clean_dirs:
        clean_paths = os.listdir(clean_dir)
        train_paths, valid_paths = train_test_split(clean_paths, 
                                                   test_size=valid_split, 
                                                   random_state=random_seed,
                                                   shuffle=True)
        settings['train_size'] += len(train_paths) * (len(transformers))
        settings['valid_size'] += len(valid_paths) * (len(transformers))

        print(f"=> Generating training set from {clean_dir}")
        for sample in tqdm(train_paths):
            clean_sample_path = os.path.join(clean_dir, sample)
            sample_audio, sr = sf.read(clean_sample_path)
            if sr != 44100:
                raise ValueError(f"Invalid sample rate: {sr}. Dataset sample rate: {settings['sample_rate']}.")
            if normalize:
                sample_audio = librosa.util.normalize(sample_audio)
            # randomly apply an fx to the sample
            if add_bypass_class:
                i_range = range(-1, len(transformers))
            else:
                i_range = range(0, len(transformers))
            for i in tqdm(i_range):
                train_clean_link.append(clean_sample_path)
                train_labels.append(transformers_labels[i])
                output_file_name = os.path.join(train_audio_dir, f"{train_sample_count}.wav")
                if i == -1:
                    sf.write(output_file_name, sample_audio, settings['sample_rate'])
                else:
                    transformers[i].build_file(input_array=sample_audio, sample_rate_in=settings['sample_rate'],
                                           output_filepath=output_file_name)
                train_sample_count += 1
        print(f"=> Generating valid set from {clean_dir}")
        for sample in tqdm(valid_paths):
            clean_sample_path = os.path.join(clean_dir, sample)
            sample_audio, sr = sf.read(clean_sample_path)
            if sr != 44100:
                raise ValueError(f"Invalid sample rate: {sr}. Dataset sample rate: {settings['sample_rate']}.")
            if normalize:
                sample_audio = librosa.util.normalize(sample_audio)
            # randomly apply an fx to the sample
            if add_bypass_class:
                i_range = range(-1, len(transformers))
            else:
                i_range = range(0, len(transformers))
            for i in tqdm(i_range):
                valid_clean_link.append(clean_sample_path)
                valid_labels.append(transformers_labels[i])
                output_file_name = os.path.join(valid_audio_dir, f"{valid_sample_count}.wav")
                if i == -1:
                    sf.write(output_file_name, sample_audio, settings['sample_rate'])
                else:
                    transformers[i].build_file(input_array=sample_audio, sample_rate_in=settings['sample_rate'],
                                           output_filepath=output_file_name)
                valid_sample_count += 1

    assert train_sample_count == settings['train_size']
    assert valid_sample_count == settings['valid_size']
    print(f"=> Generated {train_sample_count} samples to {output_full_path}/train")
    print(f"=> Generated {valid_sample_count} samples to {output_full_path}/valid")

    train_label_tensor = torch.tensor(train_labels)
    valid_label_tensor = torch.tensor(valid_labels)
    torch.save(train_label_tensor, os.path.join(f"{output_full_path}/train", "label_tensor.pt"))
    print(f"=> Training set labels written to {os.path.join(output_full_path, 'train/', 'label_tensor.pt')}")
    torch.save(valid_label_tensor, os.path.join(f"{output_full_path}/valid", "label_tensor.pt"))
    print(f"=> Validation set labels written to {os.path.join(output_full_path, 'valid/', 'label_tensor.pt')}")

    with open(os.path.join(output_full_path, 'train/', 'clean_link.csv'), 'w') as outfile:
        for sample in train_clean_link:
            outfile.write(sample)
            outfile.write('\n')
    print(f"=> Training link file written to {os.path.join(output_full_path, 'train/', 'clean_link.csv')}")
    with open(os.path.join(output_full_path, 'valid/', 'clean_link.csv'), 'w') as outfile:
        for sample in valid_clean_link:
            outfile.write(sample)
            outfile.write('\n')
    print(f"=> Validation link file written to {os.path.join(output_full_path, 'valid/', 'clean_link.csv')}")

    with open(os.path.join(output_full_path, 'settings.yml'), 'w') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False)
    print(f"=> settings written to {os.path.join(output_full_path, 'settings.yml')}")
    print("=> Done!")


def apply_fx_to_transformer(transformer, fx, fx_params):
    """
    Append the corresponding fx to the given transformer.
    """
    assert isinstance(transformer, sox.transform.Transformer)
    if fx == 'distortion':
        transformer.overdrive(**fx_params[fx]) # params: {'param1': val1, 'param2', val2}
    elif fx == 'overdrive':
        transformer.overdrive(**fx_params[fx])
    elif fx == 'feedback_delay':
        transformer.echos(**fx_params[fx])
    elif fx == 'slapback_delay':
        transformer.echo(**fx_params[fx])
    elif fx == 'reverb':
        transformer.reverb(**fx_params[fx])
    elif fx == 'chorus':
        transformer.chorus(**fx_params[fx])
    elif fx == 'flanger':
        transformer.flanger(**fx_params[fx])
    elif fx == 'phaser':
        transformer.phaser(**fx_params[fx])
    elif fx == 'tremolo':
        transformer.tremolo(**fx_params[fx])
    elif fx == 'vibrato':
        raise NotImplementedError(fx)
    elif fx == 'low_boost':
        transformer.bass(**fx_params[fx])
    elif fx == 'low_reduct':
        transformer.bass(**fx_params[fx])
    elif fx == 'mid_boost':
        transformer.equalizer(**fx_params[fx])
    elif fx == 'mid_reduct':
        transformer.equalizer(**fx_params[fx])
    elif fx == 'hi_boost':
        transformer.treble(**fx_params[fx])
    elif fx == 'hi_reduct':
        transformer.treble(**fx_params[fx])
    else:
        raise ValueError(f"Invalid effect name: {fx}")


if __name__ == "__main__":
    # data_home = "/home/jovyan/workspace/datasets/guitarset"
    # slice_guitarset(data_home=data_home, save_dir="./dataset", duration=5)
    # gen_singleFX_1on1(["dataset/clean/guitarset_5s"], "dataset/generated")
    # gen_singleFX_1onN(["dataset/clean/guitarset10"], "dataset/generated")
    # gen_singleFX_1onN(["dataset/clean/guitarset_5s"], "dataset/generated")
    # gen_multiFX(["dataset/clean/guitarset10"], "dataset/generated", [5])
    gen_multiFX(["dataset/clean/guitarset_5s"], "dataset/generated", [1, 5])
