from FrFT_code import frft
import matplotlib.pyplot as plt
import numpy as np


def specgram(f, Fs, a=[0, 1/5, 2/5, 3/5, 4/5, 5/5]):
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))

    # Random noise
    nse = 0.2 * np.random.random(size=len(t))

    for i, para in enumerate(a):
        f_frft = frft(f, para)
        if i < 3:
            ax[0][i].specgram(f_frft + nse, Fs=Fs, cmap=plt.cm.Blues)
            #ax[0][i].set_ylim(0, 1000)
            ax[0][i].grid(False)
            ax[0][i].set_title(f"a = {para}", fontsize=15)
            ax[0][i].set_xlabel('time')
            ax[0][i].set_ylabel('Hz')
        else:
            ax[1][i-3].specgram(f_frft + nse, Fs=Fs, cmap=plt.cm.Blues)
            #ax[1][i-3].set_ylim(0, 1000)
            ax[1][i-3].grid(False)
            ax[1][i-3].set_title(f"a = {para}", fontsize=15)
            ax[1][i-3].set_xlabel('time')
            ax[1][i-3].set_ylabel('Hz')

    fig.suptitle('Time/Hzuency distribution of fractional Fourier transform\n',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(pad=5)

    plt.show()


# Random seed
np.random.seed(21)

# Signal parameters
dt = 0.0005  # sample interval/spacing
t = np.arange(0.0, 20.0, dt)  # times
Fs = int(1.0 / dt)  # sampling Hzuency

# signal 1
s1 = np.sin(2 * np.pi * 400 * t)
s1[t <= 8] = s1[13 <= t] = 0

# signal 2
s2 = 1.5 * np.sin(2 * np.pi * 500 * t)
s2[t <= 8] = s2[13 <= t] = 0

# Final signal
x = s1 + s2
NFFT = 512

specgram(x, Fs=Fs)
