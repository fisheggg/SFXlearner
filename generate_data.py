import mirdata
import os
import librosa
import ddsp
import attr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm import tqdm


def slice_guitarset(data_home, track_ids, duration=5)->list:
    """
    Parameters
    ----------
    data_home: str
        path to guitarset dir.
    track_ids: list
        id of tracks in guitarset.
    duration: float
        The duration in seconds of sliced samples
    
    Returns
    -------
    guitarset_audio_sliced: list
        each element is sliced audio.
    sliced_idx: list of tuples
        each tuple in shape (track_id, slice_idx)
    
    Usage:
    import mirdata
    data_home = '/home/jovyan/workspace/datasets/guitarset'
    a, l = slice_guitarset(data_home, mirdata.guitarset.track_ids())

    """
    guitarset = mirdata.guitarset.load(data_home)
    guitarset_audio_sliced = []
    sliced_idx = []
    for track_id in tqdm(track_ids):
        audio, sr = guitarset[track_id].audio_mic
        slices = librosa.util.index_to_slice(np.arange(0, len(audio), duration*sr), idx_max=len(audio))
        for i, sli in enumerate(slices):
            if len(audio[sli]) < duration*sr:
                continue
            guitarset_audio_sliced.append(audio[sli])
            sliced_idx.append((track_id, i))
    return (guitarset_audio_sliced, sliced_idx)

@attr.s
class FXChain_sox(object):
    """
    Planning SFX:
    - Distortion: `sox.overdrive()`
    - Overdrive: `sox.gain()` or `sox.overdrive()`
    - Delay
        - Feedback delay: `sox.echo()`
        - Slapback delay: `sox.echo()`
    - Reverb: `sox.reverb()`
    - Chorus: `sox.chorus()`
    - Flanger: `sox.flanger()`
    - Phaser: `sox.phaser()`
    - Tremolo:`sox.tremolo()`
    - Vibrato: not found yet
    - EQs: `sox.equalizer()`
        - Low boost
        - Low reduct
        - Mid boost
        - Mid reduct
        - High boost
        - High reduct

    """
    effect_chain_parameter = attr.ib()
    batch_size = attr.ib(default=128)

    LIST_SUPPORT_SFX = ['Distortion',
                        'Overdrive',
                        'Feedback Delay',
                        'Slapback Delay',
                        'Reverb',
                        'Chorus',
                        'Flanger',
                        'Phaser',
                        'Tremolo',
                        'Vibrato',
                        ]

    @effect_chain_parameter.validator
    def check_support_sfx(self, attribute, value):
        for effect in value:
            assert list(effect.keys())[0] in LIST_SUPPORT_SFX, f"ERROR: Invalid effect name {effect}."

    #TODO
    def build():
        pass

    #TODO
    def build_to_file():
        pass


class FXChain_ddsp():
    
    def __init__(self, effect_chain_parameter, batch_size=128):
        """
        Args:
            effect_chain_parameter: [{effect1: {param1:val1, param2:val2, ...}}, {effect2:{param1:val1, param2:val2, ...}}, ...]
            batch_size: batch size as the input to ddsp.effects.
        """
        for effect in effect_chain_parameter:
            assert list(effect.keys())[0] in ['ExpDecayReverb', 'FilteredNoiseReverb', 'ModDelay'], "Invalid effect name: {}".format(effect)
        self.batch_size = batch_size
        self.effect_chain_parameter = effect_chain_parameter

    def _apply_ExpDecayReverb(self, batch_audio, gain, decay, reverb_length=None, **kwargs)->list:
        """
        Parameters
        ----------
        batch_audio: list of audio files.
        gain: float. parameter of ddsp.ExpDecayReverb.
        decay: float. parameter of ddsp.ExpDecayReverb.
        Returns
        -------
        output:  list, each element is effected audio.
        """
        m = len(batch_audio)
        if reverb_length is None:
            reverb_length = len(batch_audio[0])
        reverb = ddsp.effects.ExpDecayReverb(reverb_length, **kwargs)
        output_audio = reverb(batch_audio,
                                        gain=gain*np.ones([m, 1], dtype='float32'),
                                        decay=decay*np.ones([m, 1], dtype='float32'))
        return output_audio

    def build(self, input_audio)->list:
        """
        Apply effects in FXChain to input_audio.
        returns a list.
        """
        n = len(input_audio)
        effect_func = {'ExpDecayReverb': self._apply_ExpDecayReverb}
        batches = librosa.util.index_to_slice(np.arange(0, n, self.batch_size), idx_max=n)
        output_audio = None
        for batch in tqdm(batches):
            batch_output = np.array(input_audio[batch])
            for effect in self.effect_chain_parameter:
                effect_name = list(effect.keys())[0]
                batch_output = effect_func[effect_name](batch_output, **effect[effect_name])
            if output_audio is None:
                output_audio = batch_output
            else:
                output_audio = np.vstack((output_audio, batch_output))
        return output_audio

    def build_to_file(self, input_audio, track_ids, save_dir)->None:
        """
        Apply effects in FXChain to input_audio.
        returns None, and save the audio to save_dir with file name in track_ids.
        """
        if not os.path.exists(save_dir):
            print('save_dir does not exist, creating dir.')
            os.makedirs(save_dir)

        n = len(input_audio)
        effect_func = {'ExpDecayReverb': self._apply_ExpDecayReverb}
        batches = librosa.util.index_to_slice(np.arange(0, n, self.batch_size), idx_max=n)
        output_audio = None
        for batch in tqdm(batches):
            batch_output = np.array(input_audio[batch])
            for effect in self.effect_chain_parameter:
                effect_name = list(effect.keys())[0]
                batch_output = effect_func[effect_name](batch_output, **effect[effect_name])
            if output_audio is None:
                output_audio = batch_output
            else:
                output_audio = np.vstack((output_audio, batch_output))
        
        for i in tqdm(range(len(output_audio))):
            librosa.output.write_wav(os.path.join(save_dir, track_ids[i]), output_audio[i], 44100)
        
        print('All audio files saved in {}.'.format(save_dir))


# def apply_ExpDecayReverb(input_audio, batch_size=128, gain=-1., decay=1., reverb_length=None)->list:
#     """
#     Parameters
#     ----------
#     input_audio: list of audio files.
#     batch_size: int. batch size to feed in to tf when apply effect.
#     gain: float. parameter of ddsp.ExpDecayReverb.
#     decay: float. parameter of ddsp.ExpDecayReverb.

#     Returns
#     -------
#     output:  list, each element is effected audio.
#     """
#     if reverb_length is None:
#         reverb_length = len(input_audio[0])
#     reverb = ddsp.effects.ExpDecayReverb(reverb_length)
#     n = len(input_audio)
#     batches = librosa.util.index_to_slice(np.arange(0, n, batch_size), idx_max=n)
#     output_audio = None
#     for batch in tqdm(batches):
#         batch_audio = np.array(input_audio[batch])
#         m = len(batch_audio)
#         if output_audio is None:
#             output_audio = reverb(batch_audio, 
#                                                    gain=gain*np.ones([m, 1], dtype='float32'), 
#                                                    decay=decay*np.ones([m, 1], dtype='float32'))
#         else:
#             output_audio = np.vstack((output_audio, reverb(batch_audio, 
#                                                    gain=gain*np.ones([m, 1], dtype='float32'), 
#                                                    decay=decay*np.ones([m, 1], dtype='float32'))))
#     return output_audio