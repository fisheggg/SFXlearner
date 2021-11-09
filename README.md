# SFXlearner

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
|resnet18|guitarset test split|No|[1,5]|13|Mel Spectrogram|0.958|0.965||
|resnet18|guitarset test split|Yes|[1,5]|13|Mel Spectrogram|0.999|0.999||
|resnet18|IDMT-SMT-GUITAR|No|[1,5]|13|Mel Spectrogram|0.876|0.906||
|resnet18|IDMT-SMT-GUITAR|Yes|[1,5]|13|Mel Spectrogram|0.968|0.970||
|CRNN|guitarset test split|No|[1,5]|13|Mel Spectrogram|0.967|0.968|converge slow|
|CRNN|guitarset test split|yes|[1,5]|13|Mel Spectrogram|0.999|0.999|converge slow|
|CRNN|IDMT-SMT-GUITAR|No|[1,5]|13|Mel Spectrogram|0.856|0.851|converge slow|
|CRNN|IDMT-SMT-GUITAR|yes|[1,5]|13|Mel Spectrogram|0.963|0.961|converge slow|
|resnet6|IDMT-SMT-GUITAR|No|[1,5]|13|Mel Spectrogram|0.830|0.811||
|resnet6|IDMT-SMT-GUITAR|Yes|[1,5]|13|Mel Spectrogram|0.926|0.917||
|resnet10|IDMT-SMT-GUITAR|No|[1,5]|13|Mel Spectrogram|0.860|0.844||
|resnet10|IDMT-SMT-GUITAR|Yes|[1,5]|13|Mel Spectrogram|0.958|0.950||
|resnet14|IDMT-SMT-GUITAR|No|[1,5]|13|Mel Spectrogram|0.848|0.832||
|resnet14|IDMT-SMT-GUITAR|Yes|[1,5]|13|Mel Spectrogram|0.963|0.955||
