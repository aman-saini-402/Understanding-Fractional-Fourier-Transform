import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import scipy.signal
import seaborn as sns
sns.set()


def frft(f, a):
    """
    Calculate the fractional fourier transform of a function given as a
    1-D array.

    Parameters
    ----------
    f : numpy array
        The signal to be transformed.
    a : float
        fractional power
    Returns
    -------
    data : numpy array
        The transformed signal.
    References
    ---------
    ..[1] This algorith implementation is given in
          https://ieeexplore.ieee.org/document/536672
    """
    ret = np.zeros_like(f, dtype=np.complex)
    f = f.copy().astype(np.complex)
    N = len(f)
    shft = np.fmod(np.arange(N) + np.fix(N/2), N).astype(int)
    sN = np.sqrt(N)
    a = np.remainder(a, 4.0)

    # Special cases
    if a == 0.0:
        # Indentity Operator
        return f
    if a == 2.0:
        # Parity Operator
        return np.flipud(f)
    if a == 1.0:
        # Fourier Transform
        ret[shft] = np.fft.fft(f[shft]) / sN
        # ret = scipy.fftpack.fft(f)
        return ret
    if a == 3.0:
        # Inverse Fourier Transform
        ret[shft] = np.fft.ifft(f[shft]) * sN
        return ret

    # reduce to interval 0.5 < a <1.5
    if a > 2.0:
        a = a - 2.0
        f = np.flipud(f)
    elif a > 1.5:
        a = a - 1
    elif a < 0.5:
        a = a + 1
        f[shft] = np.fft.ifft(f[shft]) * sN

    # the general case for 0.5 < a < 1.5
    alpha = a * np.pi / 2
    tana2 = np.tan(alpha/2)
    sina = np.sin(alpha)

    # Shannon's Interpolation
    f = np.hstack((np.zeros(N-1), sincinterp(f), np.zeros(N-1))).T
    # chirp premultiplication
    chrp = np.exp(-1j * np.pi / N * tana2 / 4 *
                  np.arange(-2 * N + 2, 2 * N - 1).T ** 2)
    f = chrp * f

    # chirp convolution
    c = np.pi / N / sina / 4
    ret = scipy.signal.fftconvolve(
        np.exp(1j * c * np.arange(-(4 * N - 4), 4 * N - 3).T ** 2),
        f
    )
    ret = ret[4 * N - 4:8*N - 7] * np.sqrt(c / np.pi)

    # chirp post multiplication
    ret = chrp * ret

    # normalizing constant
    ret = np.exp(-1j * (1 - a) * np.pi / 4) * ret[N - 1:-N + 1:2]

    return ret


def iffrft(f, a):
    """
    Calculate the inverse fast fractional fourier transform.
    Parameters
    ----------
    f : numpy array
        The signal to be transformed.
    a : float
        fractional power
    Returns
    -------
    data : numpy array
        The transformed signal.
    """
    return frft(f, -a)


def frft2D(mat, a, b):
    m, n = mat.shape
    xvalues = np.zeros((m, n)).astype(complex)
    ret = np.zeros((m, n)).astype(complex)
    for i in range(0, m):
        xvalues[i, :] = frft(mat[i, :], a)
    for j in range(0, n):
        ret[:, j] = frft(xvalues[:, j], b)
    return ret


def sincinterp(x):
    N = len(x)
    y = np.zeros(2 * N - 1, dtype=x.dtype)
    y[:2 * N:2] = x
    xint = scipy.signal.fftconvolve(
        y[:2 * N],
        np.sinc(np.arange(-(2 * N - 3), (2 * N - 2)).T / 2),
    )
    return xint[2 * N - 3: -2 * N + 3]


def plotter(f_frft, function_name):
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))

    sns.scatterplot(range(len(f_frft[0])), (f_frft[0]), ax=ax[0][0])
    ax[0][0].plot(range(len(f_frft[0])), (f_frft[0]), color='Black')
    ax[0][0].set_title('a = 0')
    ax[0][0].set_xlabel("time")

    sns.scatterplot(range(len(f_frft[1])), abs(f_frft[1]), ax=ax[0][1])
    ax[0][1].plot(range(len(f_frft[1])), abs(f_frft[1]), color='Black')
    ax[0][1].set_title('a = 1/3')

    sns.scatterplot(range(len(f_frft[2])), abs(f_frft[2]), ax=ax[1][0])
    ax[1][0].plot(range(len(f_frft[2])), abs(f_frft[2]), color='Black')
    ax[1][0].set_title('a = 2/3')

    sns.scatterplot(range(len(f_frft[3])), abs(f_frft[3]), ax=ax[1][1])
    ax[1][1].plot(range(len(f_frft[3])), abs(f_frft[3]), color='red')
    ax[1][1].set_title('a = 1')
    ax[1][1].set_xlabel("Hz")

    fig.suptitle(
        f"Magnitudes of the fractional Fourier transforms of a {function_name}"
    )
    fig.tight_layout(pad=3)
    plt.show()


if __name__ == "__main__":
    # function and parameter choice
    # 1. rectangular function
    # 2. delta function
    # 3. constant function
    # 4. sine function
    # 5. Superposition of two sine function
    # 6. Gaussian function
    # 7. Unit step function
    # 8. sign function
    # 9. quadratic function
    i = 9

    if i == 1:
        # rectangular function
        f = np.zeros(100)
        f[40:60] = 1
        f_frft = [0]*4

        f_frft[0] = frft(f, 0)
        f_frft[1] = frft(f, 1/3)
        f_frft[2] = frft(f, 2/3)
        f_frft[3] = frft(f, 1)
        plotter(f_frft, "Rectangular Function")

    if i == 2:
        # delta function
        f = np.zeros(100)
        f[50] = 1
        f_frft = [0]*4

        f_frft[0] = frft(f, 0)
        f_frft[1] = frft(f, 1/3)
        f_frft[2] = frft(f, 2/3)
        f_frft[3] = frft(f, 1)
        plotter(f_frft, "Direc Delta Function")

    if i == 3:
        # constant function
        f = np.ones(100)
        f_frft = [0]*4

        f_frft[0] = frft(f, 0)
        f_frft[1] = frft(f, 1/3)
        f_frft[2] = frft(f, 2/3)
        f_frft[3] = frft(f, 1)
        plotter(f_frft, "Constant Function")

    if i == 4:
        # sin function
        f = np.sin(np.arange(0, 10, 0.01))
        f_frft = [0]*4

        f_frft[0] = frft(f, 0)
        f_frft[1] = frft(f, 1/3)
        f_frft[2] = frft(f, 2/3)
        f_frft[3] = frft(f, 1)
        plotter(f_frft, "Sine Function")

    if i == 5:
        # Superposition of two sine waves
        f1 = np.sin(np.arange(0, 10, 0.01))
        f2 = np.sin(np.arange(0, 10, 0.01)*50)
        f = f1 + f2
        f_frft = [0]*4

        f_frft[0] = frft(f, 0)
        f_frft[1] = frft(f, 1/3)
        f_frft[2] = frft(f, 2/3)
        f_frft[3] = frft(f, 1)
        plotter(f_frft, "Two superimposed sine waves")

    if i == 6:
        # Gaussian function
        f = np.exp(-np.pi*(np.arange(0, 5, 0.01) - 2.5)**2)
        f_frft = [0]*4

        f_frft[0] = frft(f, 0)
        f_frft[1] = frft(f, 1/3)
        f_frft[2] = frft(f, 2/3)
        f_frft[3] = frft(f, 1)
        plotter(f_frft, "Gaussian function")

    if i == 7:
        # Unit step function
        f = np.zeros(100)
        f[50:] = 1
        f_frft = [0]*4

        f_frft[0] = frft(f, 0)
        f_frft[1] = frft(f, 1/3)
        f_frft[2] = frft(f, 2/3)
        f_frft[3] = frft(f, 1)
        plotter(f_frft, "Unit step function")

    if i == 8:
        # Sign function
        f = np.zeros(100)
        f[0:50] = -1
        f[50:] = 1
        f_frft = [0]*4

        f_frft[0] = frft(f, 0)
        f_frft[1] = frft(f, 1/3)
        f_frft[2] = frft(f, 2/3)
        f_frft[3] = frft(f, 1)
        plotter(f_frft, "Sign function")

    if i == 9:
        # quadratic function
        t = np.arange(0, 10, 0.01)
        f = t*t
        f_frft = [0]*4

        f_frft[0] = frft(f, 0)
        f_frft[1] = frft(f, 1/3)
        f_frft[2] = frft(f, 2/3)
        f_frft[3] = frft(f, 1)
        plotter(f_frft, "Quadratic function")
