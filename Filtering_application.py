from FrFT_code import frft
import matplotlib.pyplot as plt
import numpy as np

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


# Noise to be removed
nse_1 = np.sin(2 * np.pi * 700 * t)
nse_1[t >= 8] = nse_1[t <= 4] = 0
nse_1 = frft(nse_1, -0.3)

# Final signal
x = s1 + s2 + nse_1
NFFT = 512


# frft
x_frft = frft(x, 0.3)

# Random noise
nse = 0.2 * np.random.random(size=len(t))


# Filtering Process
fig, ax = plt.subplots(2, 2, figsize=(14, 8))

# original Signal
ax[0][0].specgram(x + nse, Fs=Fs, cmap=plt.cm.Blues)
ax[0][0].set_ylim(0, 1000)
ax[0][0].grid(False)
ax[0][0].set_title("Signal with noise", fontsize=15)
ax[0][0].set_xlabel('time')
ax[0][0].set_ylabel('Hz')

# fractional fourier transform
ax[0][1].specgram(x_frft + nse, Fs=Fs, cmap=plt.cm.Blues)
ax[0][1].set_ylim(0, 1000)
ax[0][1].grid(False)
ax[0][1].set_title("Rotation with a=0.3", fontsize=15)
ax[0][1].set_xlabel('time')
ax[0][1].set_ylabel('Hz')

# Filtered Signal
x_filtered = frft(s1 + s2, 0.3)
ax[1][0].specgram(x_filtered + nse, Fs=Fs, cmap=plt.cm.Blues)
ax[1][0].set_ylim(0, 1000)
ax[1][0].grid(False)
ax[1][0].set_title("Rotation with a=0.3, noise is filtered", fontsize=15)
ax[1][0].set_xlabel('time')
ax[1][0].set_ylabel('Hz')

# Original filtered signal
x_filtered_frft = s1 + s2
ax[1][1].specgram(x_filtered_frft + nse, Fs=Fs, cmap=plt.cm.Blues)
ax[1][1].set_ylim(0, 1000)
ax[1][1].grid(False)
ax[1][1].set_title("Signal only", fontsize=15)
ax[1][1].set_xlabel('time')
ax[1][1].set_ylabel('Hz')

fig.suptitle('chiya the chay\n',
             fontsize=15, fontweight='bold')
fig.tight_layout(pad=5)

plt.show()
