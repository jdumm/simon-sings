# simon-sings
Ear-training game to match pitch in a Simon Says fashion.

### Overview

A quick game to see how good you can match and maintain musical pitch while humming!

Listen to the series of pitches and hum them back.  Maintain good pitch match and timing to keep your score up each round.  

### Installation

`pip install -r requirements.txt`

You may need to run `brew install portaudio` in advance to get `pyaudio` to work.

### Running

`python simon-sings.py`

Listen to the sequence of pitches.  Once you are ready to record your version, press enter.  
You will be given a score based on how well your humming matches the pure tone.  If your score
does not meet the target, you lose!

Several options are then presented to you once you lose, 
including options to see graphs of your waveforms and the constant-q transform used for scoring.

Or retry for a better score!

### Tips

The overall volume of your recording will not impact the score, as long as there is not a lot of 
background noise or if you have a bad microphone.  However, try to keep your volume constant during the
recording.

You can listen to the sum of the waveforms at the end to listen for pitch accuracy.
