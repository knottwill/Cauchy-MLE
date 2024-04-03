import numpy as np

def cauchy_pdf(x, alpha, beta):
    """
    Compute the probability density function (pdf) of Cauchy distribution at x, given location parameter alpha and scale parameter beta
    """
    return beta / (np.pi*(beta**2 + (x - alpha)**2))

def cauchy_sample(N, alpha, beta):
    """
    Generate a sample of size N from a Cauchy distribution with location parameter alpha and scale parameter beta
    
    This is done by generating uniform random numbers in (-pi/2, pi/2) and transforming them to Cauchy random numbers 
    using the equation x = alpha + beta*tan(theta), where theta is the uniform random number.
    """
    theta = np.pi*(np.random.rand(N) - 1/2) # generate uniform random numbers in (-pi/2, pi/2)
    return alpha + beta*np.tan(theta)

def cauchy_mle(x, beta, start=None, tol=1.48e-08, max_iter=1000):
    """
    Find maximum liklihood estimate (MLE) of location parameter (alpha) of Cauchy distribution, using Newton-Raphson method

    Parameters
    ----------
    x: observed dataset of Cauchy random numbers
    beta: true scale parameter of Cauchy distribution used to generate the data
    start: initial guess for location parameter alpha
    tol: tolerance for convergence
    max_iter: maximum number of iterations for Newton-Raphson method

    Returns
    -------
    mle: Maximum Likelihood Estimate of alpha
    """
    converged = False # flag to check if the algorithm has converged
    
    # define first and second derivatives of log-likelihood function
    def l1(alpha):
        """ First Derivative of log-likelihood function """
        numer = 2*(x - alpha)
        denom = beta**2 + (x - alpha)**2
        return np.sum(numer/denom)

    def l2(alpha):
        """ Second Derivative of log-likelihood function """
        numer = -2*(beta**2 - (x - alpha)**2)
        denom = (beta**2 + (x - alpha)**2)**2
        return np.sum(numer/denom)

    # Newton-Raphson method
    alpha_t = start if start is not None else np.median(x)
    for t in range(max_iter):
        alpha_t = alpha_t - l1(alpha_t)/l2(alpha_t)
        if abs(l1(alpha_t)) < tol:
            converged = True
            print(f"Newton-Raphson method converged after {t+1} iterations")
            break

    if not converged:
        print("Newton-Raphson method did not converge")
    
    return alpha_t