# SFXlearner

## Effects list

|Number| Effect name | pySoX Function name | parameters |
|:--:|:--:|:--:|:--:|
|1|overdrive|`overdrive()`|{'gain_db': 5}|
|2|distortion|`overdrive()`|{'gain_db': 15}|
|3|reverb|`reverb()`|{'reverberance': 80}|
|3|feedback_delay|`echos()`|{'n_echos': 3, 'delays': [200,400,600], 'decays':[0.4,0.2,0.1], 'gain_out':0.5}|
|4|slapback_delay| `echo()`|{'n_echos': 3, 'delays': [200,400,600], 'decays':[0.4,0.2,0.1], 'gain_out':0.5}|
|5|chorus|`chorus()`|{'n_voices': 5}|
|6|flanger|`flanger()`|{'depth': 5, 'phase': 50}|
|7|phaser|`phaser()`|{}|
|8|tremolo|`tremolo()`|{}|
|9|low_boost|`bass()`|{'frequency': 200, 'gain_db': 10}|
|10|low_reduct|`bass()`|{'frequency': 200, 'gain_db': -10}|
|11|hi_boost'|`treble()`|{'frequency': 8000, 'gain_db': 20}|
|12|hi_reduct'|`treble()`|{'frequency': 8000, 'gain_db': -20}|

## SingleFX results
|Dataset type|n_classes|Feature Type|Using Clean|Epochs|Validation Accuracy (Highest)|Notes
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|1on1|13|MFCC mean|No|100|0.41| |
|1on1|13|MFCC mean|Yes|100|0.85| |
|1onN|13|MFCC mean|No|100|0.55|converge fast|
|1onN|13|MFCC mean|Yes|100|0.96|converge fast|