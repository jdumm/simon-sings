import sys
import time
from random import choice, seed, randint

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

from ssutils import record_audio, create_pure_tone_numpy, note2freq


def do_round(n_segs=1, duration=1, sr=22050, ran_seed=1, test_mode=False):
    """ Play a series of pitches and record the user's attempt
        params:
            n_segs = Number of pitches to generate
            duration = Length of each segment in seconds
            sr = sample rate of generated and recorded audio
        returns:
            The score (0--100) for how well the user matched the generated pitches
    """
    # Start by generating randomized pure tones with sine waves for each segment
    y_segs = []
    seed(ran_seed if not test_mode else 1)
    for _ in range(n_segs):
        freq = choice(list(note2freq.values()))
        y_segs.append(create_pure_tone_numpy(freq, duration, sr))
    y = np.array(y_segs).flatten()

    y_audio_seg = AudioSegment(y.tobytes(), frame_rate=sr, sample_width=y.dtype.itemsize, channels=1)

    # Play the pure tones
    print("Now playing the tones to match...")
    play(y_audio_seg)

    # Now wait for user input to start recording their humming of the tone, or load test audio
    # TODO: add more modes?
    # modes = ['pure_single_tone', 'pure_multi_tone', 'music']
    # mode = 'pure_single_tone'
    if not test_mode:
        input("Press enter to begin recording your version!")
        time.sleep(0.01)
        outfile_name = '.user_recording_temp.wav'
        duration = y_audio_seg.duration_seconds
        x = record_audio(outfile_name=outfile_name, record_seconds=duration, chunk=1024, channels=1, rate=sr)
    else:
        audio_path = ".user_recording_default.wav"
        x, _ = librosa.load(audio_path, sr=sr)
        x = x / x.max()
        #x = np.tile(x, n_segs)
        x_audio_seg = AudioSegment(x.astype('float32').tobytes(), frame_rate=sr,
                                   sample_width=x.astype('float32').dtype.itemsize, channels=1)
        play(x_audio_seg)
    x = x[0:len(y)]  # trim recording to be the same length as the input
    y = y[0:len(x)]  # and just in case the other direction is ever needed

    # Calculate constant-q transform for y (target) and x (user recording)
    cqt_y = np.abs(librosa.cqt(y, sr=sr))
    cqt_x = np.abs(librosa.cqt(x, sr=sr))

    # Calculate user and baseline scores
    score = np.sum(np.multiply(cqt_y, cqt_x))
    baseline_score = np.sum(np.multiply(cqt_y, cqt_y))

    return score / baseline_score * 100, x, y, sr


def menu(x, y, sr):
    """ Wait for user input to decide what to do next.
    """
    response = input(
        "r) retry\nq) quit\ny) listen again to the pure tone\nx) listen to your recording\ns) listen together"
        "\ng) see graphics\n")
    if response == 'r':
        print('-- Reloading')
        main()
    elif response == 'q':
        print('-- Quiting...')
        sys.exit()
    elif response == 'y':
        print('-- Playing the pure tone')
        y_audio_seg = AudioSegment(y.astype('float32').tobytes(), frame_rate=sr,
                                   sample_width=y.astype('float32').dtype.itemsize, channels=1)
        play(y_audio_seg)
    elif response == 'x':
        print('-- Playing the user tone')
        x_audio_seg = AudioSegment(x.astype('float32').tobytes(), frame_rate=sr,
                                   sample_width=x.astype('float32').dtype.itemsize, channels=1)
        play(x_audio_seg)
    elif response == 's':
        print("-- Playing sum of waveforms")
        s = y + x
        s = s.astype('float32')  # downscale from float64 to enable playback with pydub
        s_audio_seg = AudioSegment(s.tobytes(), frame_rate=sr, sample_width=y.dtype.itemsize, channels=1)
        play(s_audio_seg)
    elif response == 'g':
        print('-- Plotting waveform and constant-q transform')
        # Display the waveforms
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 9))
        plt.tight_layout(pad=2.0)
        ax0.set_title("Full Audio")
        librosa.display.waveplot(y, sr=sr, ax=ax0, color='k', alpha=0.3, label='Pure')
        librosa.display.waveplot(x, sr=sr, ax=ax0, color='b', alpha=0.5, label='You')
        ax0.legend(fontsize=16)
        ax0.set_xlabel('')

        ax1.set_title("Sum of Waveforms")
        librosa.display.waveplot(y+x, sr=sr, ax=ax1, color='orange', alpha=0.5, label='Sum')
        ax1.legend(fontsize=16)
        ax1.set_xlabel('')

        frac = 0.05
        offset = int(min(x.argmax(), len(x) - frac * len(x)))
        ax2.set_title(f"{frac * 100:0.0f}% Sample Near Max Volume")
        librosa.display.waveplot(y[offset:offset + int(frac * len(y))], sr=sr, ax=ax2, color='k', alpha=0.3,
                                 label='Pure')
        librosa.display.waveplot(x[offset:offset + int(frac * len(x))], sr=sr, ax=ax2, color='b', alpha=0.5,
                                 label='You')
        plt.draw()

        # Calculate constant-q transform for y (target) and x (user recording) and display them
        cqt_y = np.abs(librosa.cqt(y, sr=sr))
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 9))
        plt.tight_layout(pad=2.0)
        db_vs_time_y = librosa.amplitude_to_db(cqt_y, ref=np.min)
        img_y = librosa.display.specshow(db_vs_time_y,
                                         sr=sr, x_axis='time', y_axis='cqt_note', ax=ax0)
        ax0.set_title('Constant-Q Power Spectrum - Pure Tones')
        ax0.set_xlabel('')
        fig.colorbar(img_y, ax=ax0, format="%+2.0f dB")

        cqt_x = np.abs(librosa.cqt(x, sr=sr))
        db_vs_time_x = librosa.amplitude_to_db(cqt_x, ref=np.min)
        img_x = librosa.display.specshow(db_vs_time_x, sr=sr, x_axis='time', y_axis='cqt_note', ax=ax1)
        ax1.set_title('Constant-Q Power Spectrum - Recorded Tones')
        ax1.set_xlabel('')
        fig.colorbar(img_x, ax=ax1, format="%+2.2f dB")

        img_x = librosa.display.specshow(cqt_y * cqt_x, sr=sr, x_axis='time', y_axis='cqt_note', ax=ax2)
        ax2.set_title('Product (Overlap) of Power Spectra')
        fig.colorbar(img_x, ax=ax2, format="%+2.0f dB")
        plt.draw()

        print(' * Close the graphs to proceed')
        plt.show()
    else:
        print('-- Input not understood')
    menu(x, y, sr)


def main():
    """ Execute Simon Sings game.
    """
    test_mode = False # Set to True to use a default recording and fixed random seed - for testing
    sr = 22050  # sample rate
    n_round = 0
    n_segs = 3  # Initial number of tone segments
    ran_seed = randint(0, 2**32-1)  # Finite set of games!
    difficulties = [5, 10, 15]  # TODO: Add difficulty selector, easy/medium/hard
    target_score = difficulties[1]  # Index 0 = Easy mode
    current_score = 100
    sum_score = 0
    while current_score >= target_score:
        n_round = n_round + 1
        print(f"Beginning round number {n_round}.  {n_segs} pitches to match!")

        current_score, x, y, sr = do_round(n_segs, duration=1, sr=sr, ran_seed=ran_seed, test_mode=test_mode)

        print(f"Target score is:  {target_score:0.1f}")
        print(f"Current score is: {current_score:0.1f}")

        n_segs = n_segs + 1
        sum_score = sum_score + current_score
    else:
        print(f" *** You made it {n_round} rounds with final score: {sum_score:0.1f}! ***")
        menu(x, y, sr)


if __name__ == '__main__':
    main()
