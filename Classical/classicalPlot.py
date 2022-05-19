import numpy as np
import matplotlib.pyplot as plt

def plot(K, N=200, r=.5):
    """ This comes from @t-makaro program on Github, alongside another code part called 'iteration' """
    Q, P, colours = iteration(K, N, n)

    plt.scatter(Q,P, s=.1, c=colours) # A scatter plot of y vs. x with varying marker size and/or color. s is the marker size 
    plt.xlabel('Q')
    plt.ylabel('P')
    plt.show()


def iteration_density(K, sample, nkick=300, N=100, r=1):
    """ The sample is a list of the Kick numbers the user wants to plot, the code below is a 
        very personal version, and you may want to change the value of the colours or add more possible colours.
        It was initally designed for a sample length of 4."""    
    
    p = np.linspace(-10,10,N)
    x = np.zeros(N)
    pmax = 300
    coeff = np.zeros((nkick,2))
    
    colours1 = ['#1f77b4', '#17becf', '#9467bd', '#e377c2', '#bcbd22']
    colours2 = ['#19224f', '#089c4a', '#482d7d', '#a13097', '#bd6d22']
    
    index = 0
    
    fig = plt.figure()
    fig.suptitle(f'Probability Density \n in the \n Phase Space (K={K})', fontsize=20)
    plt.xlabel('P')
    plt.ylabel('density')
    for i in range(nkick):
        if i in sample:
            
            n, bins,_ = plt.hist(p, range=(-pmax,pmax), bins=int(np.floor(2*pmax/r)), density=True, color=colours1[index],  label=f'After {i} Kicks')
            mu, sigma = norm.fit(p)
            coeff[i] = sigma
            best_fit_line = norm.pdf(bins, mu, sigma)
            plt.plot(bins, best_fit_line, color=colours2[index], label=f'sigma={round(sigma,2)}')
            index +=1
            
        for j in range(N):
            p[j] += K*np.sin(x[j])
            x[j] += p[j]  
            
    plt.legend(ncol=2,handleheight=2.4, labelspacing=0.05)
    plt.show()
    return coeff
