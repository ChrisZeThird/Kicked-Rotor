from numpy import pi, linspace, zeros, floor, array, arange, sin
import matplotlib.pyplot as plt
from scipy.stats import norm

from Classical.classicalKickedRotor import ClassicalKickedRotor

CKR = ClassicalKickedRotor()


def iteration_density(K: float, sample: list, nkick=300, N=100, r=1.):
    """ The sample is a list of the Kick numbers the user wants to plot, the code below is a
        very personal version, and you may want to change the value of the colours or add more possible colours.
        It was initially designed for a sample length of 4.
    Input:
        K -> int, width of the studied space
        sample -> list, list of the kick numbers to plot
        nkick -> int, Number of kick
        N -> int, width of the studied space
        r -> float, resolution of the histogram
    """

    p = linspace(start=-10, stop=10, num=N)
    x = zeros(N)
    pmax = 300
    coeff = zeros((nkick, 2))

    colours1 = ['#1f77b4', '#17becf', '#9467bd', '#e377c2', '#bcbd22']
    colours2 = ['#19224f', '#089c4a', '#482d7d', '#a13097', '#bd6d22']

    index = 0

    fig = plt.figure()
    fig.suptitle(f'Probability Density \n in the \n Phase Space (K={K})', fontsize=20)
    plt.xlabel('P')
    plt.ylabel('density')
    for i in range(nkick):
        if i in sample:

            n, bins, _ = plt.hist(p, range=(-pmax, pmax), bins=int(floor(2*pmax/r)), density=True, color=colours1[index],  label=f'After {i} Kicks')
            mu, sigma = norm.fit(p)
            coeff[i] = sigma
            best_fit_line = norm.pdf(bins, mu, sigma)
            plt.plot(bins, best_fit_line, color=colours2[index], label=f'sigma={round(sigma,2)}')
            index += 1

        for j in range(N):
            p[j] += K*sin(x[j])
            x[j] += p[j]

    plt.legend(ncol=2, handleheight=2.4, labelspacing=0.05)
    plt.show()
    return coeff


if __name__ == "__main__":
    K_arr = array([0, 0.1, 0.3, 10])
    r = 0.5

    p0 = arange(start=0, stop=2 * pi, step=r)
    q0 = arange(start=0, stop=2 * pi, step=r)

    fig = plt.figure()
    fig.suptitle('Chirikov Standard Map \n Phase Space of the Kicked Rotor')

    for i in range(len(K_arr)):
        P, Q, colours = CKR.iteration(q0=q0, p0=p0, K=K_arr[i], n=200)

        ax = fig.add_subplot(2, 2, i + 1)
        ax.scatter(Q, P, s=.1,
                   c=colours)  # A scatter plot of y vs. x with varying marker size and/or color. s is the marker size
        ax.set_xlabel('Q')
        ax.set_ylabel('P')
        ax.set_title(f'K={K_arr[i]}')

    plt.tight_layout()
    plt.show()
