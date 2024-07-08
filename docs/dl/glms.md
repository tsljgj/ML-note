# Generalized Linear Models

## The Exponential Family
We have already discussed regression problem and classification problem. Note that we assume, for example, in linear regression that the noise follow a Gaussian distribution, or in logistic regression that $y$ is a Bernoulli distribution. However, not every data set is close to Gaussian distribution. It may be a better choice to choose some other distribution, such as Poisson distribution. After we change the distribution, we still need to get to its derivative or likelihood view. In this chapter, we are looking for a generalized way of viewing linear models. With this tech, we can shift to different distribution easily.

!!! df "**Definition** (The Exponential Family)"
    A class of distributions is in the exponential family if it can be written in the form 
    
    $$\begin{align*}
    p(y;\eta) = b(y)\text{exp}(\eta^TT(y)-a(\eta))
    \end{align*}$$
    
    Here, $\eta$ is called the **natural parameter** (also called the **canonical parameter**) of the distribution; $T(y)$ is the **sufficient  statistic**, which we normally set to be its identity function, i.e., $T(y) = y$; and $a(\eta)$ is the **log partition function**. $e^{a(\eta)}$ essentially plays the role of a normalization constant, that make sure the distribution $p(y;\eta)$ sums/integrates over $y$ to $1$. <br>
    
    A fixed choice of T, a and b defines a family (or set) of distributions that
    is parameterized by η; as we vary η, we then get different distributions within
    this family.

!!! eg "**Example** (Bernoulli Distribution)"
    We write the Bernoulli distribution as:

    $$
    \begin{align*}
        p(y; \phi) &= \phi^y (1 - \phi)^{1-y} \\
                &= \exp(y \log \phi + (1 - y) \log (1 - \phi)) \\
                &= \exp \left( \left( \log \left( \frac{\phi}{1 - \phi} \right) \right) y + \log (1 - \phi) \right).
    \end{align*}
    $$

    Thus, the natural parameter is given by \(\eta = \log(\phi / (1 - \phi))\). Interestingly, if we invert this definition for \(\eta\) by solving for \(\phi\) in terms of \(\eta\), we obtain \(\phi = 1 / (1 + e^{-\eta})\). This is the familiar sigmoid function! This will come up again when we derive logistic regression as a GLM. To complete the formulation of the Bernoulli distribution as an exponential family distribution, we also have

    $$
    \begin{align*}
        T(y) &= y \\
        a(\eta) &= -\log(1 - \phi) \\
                &= \log(1 + e^{\eta}) \\
        b(y) &= 1
    \end{align*}
    $$

    This shows that the Bernoulli distribution belongs to the exponential family.

!!! eg "**Example** (Gaussian Distribution)"
    Let's now move on to consider the Gaussian distribution. Recall that, when deriving linear regression, the value of \(\sigma^2\) had no effect on our final choice of \(\theta\) and \(h_{\theta}(x)\). Thus, we can choose an arbitrary value for \(\sigma^2\) without changing anything. To simplify the derivation below, let's set \(\sigma^2 = 1\). We then have:

    $$
    \begin{align*}
        p(y; \mu) &= \frac{1}{\sqrt{2\pi}} \exp \left( -\frac{1}{2}(y - \mu)^2 \right) \\
                &= \frac{1}{\sqrt{2\pi}} \exp \left( -\frac{1}{2} y^2 \right) \cdot \exp \left( \mu y - \frac{1}{2} \mu^2 \right)
    \end{align*}
    $$

    Thus, we see that the Gaussian is in the exponential family, with

    $$
    \begin{align*}
        \eta &= \mu \\
        T(y) &= y \\
        a(\eta) &= \mu^2 / 2 \\
                &= \eta^2 / 2 \\
        b(y) &= \left( \frac{1}{\sqrt{2\pi}} \right) \exp \left( -y^2 / 2 \right)
    \end{align*}
    $$

    There're many other distributions that are members of the exponential family: The multinomial (which we'll see later), the Poisson (for modelling count-data; also see the problem set); the gamma and the exponential (for modelling continuous, non-negative random variables, such as time-intervals); the beta and the Dirichlet (for distributions over probabilities); and many more. 

## Constructing GLMs
- [ ] Constructing GLMs
- [ ] Least Square Revisit
- [ ] Logistic Regression Revisit