# SFXlearner

## Effects list

|Number| Effect name | pySoX Function name | parameters |
|:--:|:--:|:--:|:--:|
|1|overdrive|`overdrive()`|{'gain_db': 5}|
|2|distortion|`overdrive()`|{'gain_db': 15}|
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
