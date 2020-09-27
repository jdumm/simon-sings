import wave

import numpy as np
from pyaudio import PyAudio, paInt16


def record_audio(outfile_name=None, record_seconds=1, chunk=1024, channels=1, rate=22050):
    """ Opens microphone (OS may ask for permission) and records <record_seconds> if audio.
        If an outfile_name is provided, recording is saved into the file.
        returns: A normalized numpy array of the waveform.
    """

    format_ = paInt16
    p = PyAudio()

    stream = p.open(format=format_,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    print("* Recording Audio")

    frames_for_file = []
    frames_for_np = []

    mic_fudge = 1  # in chunks, delay start of recording to avoid dead time
    for i in range(0, int(rate / chunk * record_seconds) + mic_fudge + 1):
        if i < mic_fudge:  # discard first chunk(s)
            stream.read(chunk)
            continue
        data = stream.read(chunk)
        frames_for_file.append(data)
        frames_for_np.append(np.frombuffer(data, dtype=np.int16))

    print("* Done Recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    if outfile_name is not None:
        with wave.open(outfile_name, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format_))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames_for_file))
            wf.close()

    frames_for_np = np.array(frames_for_np).flatten()
    return frames_for_np / np.max(frames_for_np)


def create_pure_tone_numpy(freq, duration, sr):
    """Generate a pure tone at <freq> Hz and <duration> seconds at <sr> Sampling Rate
    """
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)  # time grid
    y = 1.0 * np.sin(2 * np.pi * freq * t)  # pure sine wave at <freq> Hz
    return y.astype('float32')


note2freq = {
    "A2": 110.00,
    "A#2": 116.54,
    "B2": 123.47,
    "C3": 130.81,
    "C#3": 138.59,
    "D3": 146.83,
    "D#3": 155.56,
    "E3": 164.81,
    "F3": 174.61,
    "F#3": 185.00,
    "G3": 196.00,
    "G#3": 207.65,
    "A3": 220.00,
}
