# SFXlearner

[![DOI](https://zenodo.org/badge/344083237.svg)](https://zenodo.org/badge/latestdoi/344083237)
<a target="_blank" href="https://colab.research.google.com/github/fisheggg/SFXlearner/blob/main/playground.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This is the codebase for paper "Automatic Recognition of Cascaded Guitar Effects" at DAFX23.

A bunch of rendered examples can be found here: https://github.com/fisheggg/SFXlearner/tree/main/dataset/rendered/00_BN1-129-Eb_comp_1.wav 

## Effects list

|Number| Effect name | pySoX Function name | parameters |
|:--:|:--:|:--:|:--:|
|1|overdrive|`overdrive()`|{'gain_db': 5}|
|2|distortion|`overdrive()`|{'gain_db': 15}|
|3|chorus|`chorus()`|{'n_voices': 5}|
|4|flanger|`flanger()`|{'depth': 5, 'phase': 50}|
|5|phaser|`phaser()`|{}|
|6|tremolo|`tremolo()`|{}|
|7|reverb|`reverb()`|{'reverberance': 80}|
|8|feedback_delay|`echos()`|{'n_echos': 3, 'delays': [200,400,600], 'decays':[0.4,0.2,0.1], 'gain_out':0.5}|
|9|slapback_delay| `echo()`|{'n_echos': 3, 'delays': [200,400,600], 'decays':[0.4,0.2,0.1], 'gain_out':0.5}|
|10|low_boost|`bass()`|{'frequency': 200, 'gain_db': 10}|
|11|low_reduct|`bass()`|{'frequency': 200, 'gain_db': -10}|
|12|hi_boost'|`treble()`|{'frequency': 8000, 'gain_db': 20}|
|13|hi_reduct'|`treble()`|{'frequency': 8000, 'gain_db': -20}|

## SingleFX results
|Dataset type|n_classes|Feature Type|Using Clean|Epochs|Validation Accuracy (Highest)|Notes
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|1on1|13|MFCC mean|No|100|0.41| |
|1on1|13|MFCC mean|Yes|100|0.85| |
|1onN|13|MFCC mean|No|100|0.55|converge fast|
|1onN|13|MFCC mean|Yes|100|0.96|converge fast|

## MultiFX results
|Model type|Test set|With clean|effect method|n_classes|Feature Type|micro F1|macro F1|Notes|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|CRNN|guitarset test split|Yes|[1,5]|13|Mel Spectrogram|0.999|0.999|converge slow|
|resnet18|guitarset test split|Yes|[1,5]|13|Mel Spectrogram|0.999|0.999||
|baseline|guitarset test split|Yes|[1,5]|13|MFCC|0.951|0.952|MFCC+MLP|
|sampleCNN|guitarset test split|Yes|[1,5]|13|Raw audio|0.864|0.684||
||||||||||
|CRNN|IDMT-SMT-GUITAR|Yes|[1,5]|13|Mel Spectrogram|0.963|0.961|converge slow|
|resnet18|IDMT-SMT-GUITAR|Yes|[1,5]|13|Mel Spectrogram|0.968|0.970||
|resnet14|IDMT-SMT-GUITAR|Yes|[1,5]|13|Mel Spectrogram|0.963|0.955||
|resnet10|IDMT-SMT-GUITAR|Yes|[1,5]|13|Mel Spectrogram|0.958|0.950||
|resnet6|IDMT-SMT-GUITAR|Yes|[1,5]|13|Mel Spectrogram|0.926|0.917||
|baseline|IDMT-SMT-GUITAR|Yes|[1,5]|13|MFCC|0.779|0.772|MFCC+MLP|
|sampleCNN|IDMT-SMT-GUITAR|Yes|[1,5]|13|Raw audio|0.804|0.640||
||||||||||
||||||||||
|CRNN|guitarset test split|No|[1,5]|13|Mel Spectrogram|0.967|0.968|converge slow|
|resnet18|guitarset test split|No|[1,5]|13|Mel Spectrogram|0.958|0.965||
|baseline|guitarset test split|No|1,5]|13|MFCC|0.892|0.897|MFCC+MLP|
|sampleCNN|guitarset test split|No|[1,5]|13|Raw audio|0.877|0.778||
||||||||||
|CRNN|IDMT-SMT-GUITAR|No|[1,5]|13|Mel Spectrogram|0.856|0.851|converge slow|
|resnet18|IDMT-SMT-GUITAR|No|[1,5]|13|Mel Spectrogram|0.876|0.906||
|resnet14|IDMT-SMT-GUITAR|No|[1,5]|13|Mel Spectrogram|0.848|0.832||
|resnet10|IDMT-SMT-GUITAR|No|[1,5]|13|Mel Spectrogram|0.860|0.844||
|resnet6|IDMT-SMT-GUITAR|No|[1,5]|13|Mel Spectrogram|0.830|0.811||
|baseline|IDMT-SMT-GUITAR|No|[1,5]|13|MFCC|0.704|0.696|MFCC+MLP|
|sampleCNN|IDMT-SMT-GUITAR|No|[1,5]|13|Raw audio|0.697|0.623||
