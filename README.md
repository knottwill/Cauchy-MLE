# Cauchy MLE

Maximum likelihood estimation (MLE) of the location parameter of the Cauchy (Lorentzian) distribution.

## Usage

All functionality is contained inside `cauchy.py`. Example usage:

```python
from cauchy import cauchy_sample, cauchy_mle

alpha = 2 # location parameter
beta = 1  # scale parameter

# generate random sample of size 10000 from Cauchy distribution
x = cauchy_sample(10000, alpha, beta)

# Maximum likelihood estimation of location parameter
alpha_mle = cauchy_mle(x=x,            # sample
                       beta=beta,      # known scale parameter
                       start=1,        # initial guess (if None, we use median of x)
                       tol=1e-6,       # tolerance for convergence
                       max_iter=1000)  # maximum number of iterations
```

## Theory

#### Cauchy Distribution

$$\text{Cauchy}(x; \alpha, \beta) = \frac{\beta}{\pi(\beta^2 + (x - \alpha)^2)},$$

where $\alpha$ is the location parameter and $\beta$ is the scale parameter.

#### Maximum Likelihood Estimation

For a set of independent Cauchy random variables $\{x_k\}_{k=1}^N$, the likelihood function is given by

$$
L(x|\alpha, \beta) = \prod_{k=1}^N \frac{\beta}{\pi(\beta^2 + (x_k - \alpha)^2)}
$$

The log-likelihood function is then 

$$
l = \ln L(x|\alpha, \beta) = N\ln{\frac{\beta}{\pi}} - \sum_{k=1}^N\ln{(\beta^2 + (x_k - \alpha)^2)}.
$$

The first derivative with respect to $\alpha$ is

$$
l' = \frac{\partial l}{\partial\alpha} = \sum_{k=1}^{N}\frac{2(x_k - \alpha)}{\beta^2 + (x_k - \alpha)^2}
$$

and the second derivative is

$$
l'' = \frac{{\partial^2}l}{{\partial\alpha^2}} = -2\sum_{k=1}^{N}\frac{\beta^2 - (x_k - \alpha)^2}{(\beta^2 + (x_k - \alpha)^2)^2}.
$$

We typically obtain the maximum likelihood estimate by setting $l'(\hat{\alpha}) = 0$, but in the case of the Cauchy distribution, there is no closed form expression for $\hat{\alpha}$, thus we estimate it using the Newton-Raphson method. We pick an initial guess, $\alpha_0$, then iterate the following expression until convergence:

$$
\alpha_{t+1} = \alpha_{t} - \frac{l'(\alpha_{t})}{l''(\alpha_{t})}.
$$

#### Sampling

To sample from $\text{Cauchy}(x; \alpha, \beta)$, we can sample $\theta \in (-\frac{\pi}{2}, \frac{\pi}{2})$ uniformly, and transform it using the equation $x = \alpha + \beta \tan{\theta}$. 

<b>Proof:</b> Let $x = \alpha + \beta \tan{\theta}$, where $\theta$ is a uniformly distributed random variable over $(-\frac{\pi}{2}, \frac{\pi}{2})$. This equation for $x$ is monotonically increasing over $\theta \in (-\frac{\pi}{2}, \frac{\pi}{2})$, hence the PDF of $x$ satisfies $f(x)dx = u(\theta)d\theta$, where $u(\theta)$ is the PDF of $\theta$, given by 

$$
u(\theta) = 
\begin{cases} 
\frac{1}{\pi} & \text{for } \theta \in (-\frac{\pi}{2}, \frac{\pi}{2}), \\
0 & \text{otherwise}.
\end{cases}
$$

Then, $f(x) = u(\theta)|\frac{d\theta}{dx}| = \frac{1}{\pi}|\frac{d\theta}{dx}| = \frac{\beta}{\pi(\beta^2 + (x - \alpha)^2)} = \text{Cauchy(x)}$

## Credits

https://utstat.toronto.edu/keith/papers/cauchymle.pdf
