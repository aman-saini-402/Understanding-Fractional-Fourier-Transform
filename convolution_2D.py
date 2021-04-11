from FrFT_code import frft2D
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

# Random seed
np.random.seed(21)


def verify_fractional_convolution2D(f_mat, g_mat, x_para, y_para):
    alpha = x_para*np.pi/2
    beta = y_para*np.pi/2
    gamma = (alpha+beta)/2
    # omega = (alpha-beta)/2
    a = np.complex(1j * (1 / (2 * np.tan(gamma))))
    # b = np.complex(1j * np.cos(omega) / np.sin(gamma))
    # c = np.complex(1j * np.sin(omega) / np.sin(gamma))
    d = np.complex(1j * np.exp(-1j * gamma) / (2 * np.pi * np.sin(gamma)))
    points = np.arange(0, f_mat.shape[0])
    x_mat = np.array([[list(points)]*len(points)][0]).T
    y_mat = np.array([[list(points)]*len(points)][0])
    f_hat = np.exp(-a * (x_mat**2 + y_mat**2)) * f_mat
    g_hat = np.exp(-a * (x_mat**2 + y_mat**2)) * g_mat

    conv = scipy.signal.fftconvolve(f_hat, g_hat)  # , mode= 'same')
    N = x_mat.shape[0]
    h_mat = d * np.exp(a * (x_mat**2 + y_mat**2)) * conv[0:2*N:2, 0:2*N:2]
    h_mat = h_mat * np.exp(-1j * (1 - gamma) * np.pi / 4)

    # fractional fourier transforms
    Fa_mat = frft2D(f_mat, x_para, y_para)
    Ga_mat = frft2D(g_mat, x_para, y_para)
    Ha_mat = frft2D(h_mat, x_para, y_para)

    rhs = Fa_mat * Ga_mat * np.exp(a * (x_mat**2 + y_mat**2))

    # PLOTS
    X, Y = np.meshgrid(points, points)
    # Plots of the original functions and there Fourier transforms
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(241, projection='3d')
    ax1.plot_surface(X, Y, f_mat)
    ax1.set_xlabel('(a)')

    ax2 = fig.add_subplot(242, projection='3d')
    ax2.plot_surface(X, Y, g_mat)
    ax2.set_xlabel('(b)')

    ax3 = fig.add_subplot(243, projection='3d')
    ax3.plot_surface(X, Y, abs(Fa_mat))
    ax3.set_xlabel('(c)')

    ax4 = fig.add_subplot(244, projection='3d')
    ax4.plot_surface(X, Y, abs(Ga_mat))
    ax4.set_xlabel('(d)')

    points_h = np.arange(0, h_mat.shape[0])
    X_h, Y_h = np.meshgrid(points_h, points_h)
    ax5 = fig.add_subplot(245, projection='3d')
    ax5.plot_surface(X, Y, abs(h_mat), color='red')
    ax5.set_xlabel('(e)')

    ax6 = fig.add_subplot(246, projection='3d')
    ax6.plot_surface(X, Y, abs(Ha_mat), color='red')
    ax6.set_xlabel('(f)')

    ax7 = fig.add_subplot(247, projection='3d')
    ax7.plot_surface(X, Y, abs(rhs))
    ax7.set_xlabel("(g)")

    fig.suptitle("Verification of Fractional Convolution in 2D", fontsize=20)
    fig.tight_layout(pad=3)
    plt.show()

    # convolution of f and g
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, abs(h_mat))
    # ax.set_title("h(x,y)")
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # fig.suptitle("Convolution of f and g")
    # plt.show()
    #
    # # Verification of the fractional convolution theorem
    # fig = plt.figure(figsize=(12, 6))
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.plot_surface(X, Y, abs(Ha_mat))
    # ax1.set_title("Fractional Fourier transform of h(x,y)")
    #
    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.plot_surface(X, Y, abs(rhs))
    # ax2.set_title("RHS of the expression that is supposed to be equal to FrFT of h(x,y)")
    # fig.tight_layout(pad=3)
    # plt.show()


if __name__ == "__main__":
    # signal 1
    # 1. Rectangular window
    # 2. 2D Sine wave
    # 3. Constant function

    # signal 2
    # 1. Rectangular window
    # 2. 2D Sine wave
    # 3. Constant function

    s1 = 3
    s2 = 2
    # 2D FRFT Parameters
    x_para = 0.9
    y_para = 0.99

    # Define the space
    x = np.arange(0, 20)
    y = np.arange(0, 20)
    X, Y = np.meshgrid(x, y)

    if s1 == 1:
        f = np.zeros((len(x), len(y)))
        f[np.argmax(x == 6):np.argmax(x == 14), np.argmax(y == 6):np.argmax(y == 14)] = 1
    if s1 == 2:
        f = 1.5*np.sin(np.sqrt(X ** 2 + Y ** 2))
    if s1 == 3:
        f = np.ones((len(x), len(y)))

    if s2 == 1:
        g = np.zeros((len(x), len(y)))
        g[np.argmax(x == 6):np.argmax(x == 14), np.argmax(y == 6):np.argmax(y == 14)] = 1
    if s2 == 2:
        g = 2*np.sin(np.sqrt(X ** 2 + Y ** 2))
    if s2 == 3:
        g = np.ones((len(x), len(y)))

    # 2D Fractional Convolution
    verify_fractional_convolution2D(f, g, x_para, y_para)
