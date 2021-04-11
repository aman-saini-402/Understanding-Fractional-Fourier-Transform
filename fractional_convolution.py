import matplotlib.pyplot as plt
from FrFT_code import frft, sincinterp
import numpy as np
import scipy.signal

# Random seed
np.random.seed(21)


def verify_fractional_convolution(f, g, para):
    alpha = para*np.pi/2
    a = 1/(2*np.tan(alpha))
    # b = 1/np.cos(alpha)
    c = np.complex(np.sqrt(1 - 1j/np.tan(alpha)))
    f = np.array(f).astype(np.complex)
    g = np.array(g).astype(np.complex)
    # t = np.arange(0, len(f)*dt, dt)

    # FRACTIONAL CONVOLUTION
    tana2 = np.tan(alpha/2)
    sina = np.sin(alpha)
    N = len(f)
    f2 = np.hstack((np.zeros(N-1), sincinterp(f), np.zeros(N-1))).T
    g2 = np.hstack((np.zeros(N-1), sincinterp(g), np.zeros(N-1))).T
    chrp = np.exp(-1j * np.pi / N * tana2 / 4 *
                  np.arange(-2 * N + 2, 2 * N - 1).T ** 2)
    f_hat = f2 * chrp
    g_hat = g2 * chrp
    # f_hat = f * np.exp(1j * a * t**2)
    # g_hat = f * np.exp(1j * a * t**2)

    conv = np.convolve(f_hat, g_hat)
    # t_conv = np.arange(0, 2 * len(f) * dt - dt, dt)

    h = (c / np.sqrt(2 * np.pi)) * chrp * \
        conv[int(len(conv)/2 - len(chrp)/2):int(len(conv)/2 + len(chrp)/2)]
    # h = np.exp(-1j * (1 - para) * np.pi / 4) * \
    # conv[int(len(conv)/2 - len(chrp)/2):int(len(conv)/2 + len(chrp)/2)]
    # h = (c / np.sqrt(2 * np.pi)) * np.exp(-1j * a * t**2) * conv

    # fractional fourier transforms
    Fa = frft(f, para)
    Ga = frft(g, para)
    Ha = frft(h, para)

    rhs = Fa * Ga * np.exp(-1j * a * np.arange(0, len(f))**2)
    # rhs = Fa * Ga
    # rhs = np.hstack((np.zeros(N-1), sincinterp(rhs), np.zeros(N-1))).T
    # rhs = rhs * chrp

    # Plots
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))

    ax[0, 0].plot(f)
    ax[0, 0].set_xlabel("(a)")
    #ax[0, 0].set_title("f(t)", fontsize=15)

    ax[0, 1].plot(g)
    ax[0, 1].set_xlabel("(b)")
    #ax[0, 1].set_title("g(t)", fontsize=15)

    ax[0, 2].plot(abs(Fa))
    ax[0, 2].set_xlabel("(c)")
    #ax[1, 0].set_title(f"FrFT of f(t) with a = {para}")

    ax[0, 3].plot(abs(Ga))
    ax[0, 3].set_xlabel("(d)")
    #ax[1, 1].set_title(f"FrFT of g(t) with a = {para}")

    ax[1, 0].plot(abs(h), color='red')
    ax[1, 0].set_xlabel("(e)")

    ax[1, 1].plot(abs(Ha), color='red')
    ax[1, 1].set_xlabel("(f)")

    ax[1, 2].plot(abs(rhs))
    ax[1, 2].set_xlabel("(g)")

    fig.delaxes(ax[1, 3])

    fig.suptitle("Verification of Fractional Convolution", fontsize=20)
    fig.tight_layout(pad=3)

    plt.show()

    # Plots to verify the convolution Identity
    # plt.plot(abs(h), color='red')
    # plt.title("Fractional Convolution of f(t) and g(t)")
    # plt.xlabel("time")
    # plt.show()
    #
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    #
    # ax[0].plot(abs(Ha), color='red')
    # ax[0].set_title(f"Fractional Fourier transform of h(t) with a = {para}", fontsize=15)
    #
    # ax[1].plot(abs(rhs))
    # ax[1].set_title(
    #     f"RHS of the expression that is supposed to be equal to FrFT of h(t)", fontsize=15)
    #
    # fig.tight_layout(pad=3)
    # plt.show()


if __name__ == "__main__":
    # signal 1
    # 1. Sine wave
    # 2. Constant function
    # 3. rectangular function
    # 4. Delta function

    # signal 2
    # 1. Sine wave
    # 2. Constant function
    # 3. rectangular function
    # 4. Delta function

    # Set these parameters
    s1 = 2
    s2 = 4
    # Frft parameter
    a = 0.25
    # Signal parameters
    dt = 0.005  # sample interval/spacing
    Fs = int(1.0 / dt)  # sampling Hzuency
    t = np.arange(0, 20, dt)  # sampling time

    if s1 == 1:
        f = np.sin(2 * np.pi * 0.2 * t)
    if s1 == 2:
        f = [1]*len(t)
    if s1 == 3:
        f = np.zeros(len(t))
        f[int(len(t)/4):int(3*len(t)/4)] = 1
    if s1 == 4:
        f = np.zeros_like(len(t))
        f[int(len(t)/2)] = 1

    if s2 == 1:
        g = 1.5 * np.sin(2 * np.pi * 0.7 * t)
    if s2 == 2:
        g = [1]*len(t)
    if s2 == 3:
        g = np.zeros(len(t))
        g[int(len(t)/4):int(3*len(t)/4)] = 1
    if s2 == 4:
        g = np.zeros_like(t)
        g[int(len(t)/2)] = 1

    # Fractional Convolution
    verify_fractional_convolution(f, g, a)
