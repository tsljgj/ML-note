# Linear Regression

## Loss Function
!!! im "**Important Note** (Probabilistic Interpretation of Squared Loss)"
    Why squared loss is a reasonable choice? Assume our target $y^{(i)}$, feature vector $x^{(i)}$, weights vector $\theta$, and bias $\epsilon^{(i)}$ are related via the equation:
    
    $$\begin{align*}
    y^{(i)} = \theta^{T}x^{(i)} + \epsilon^{(i)}
    \end{align*}$$

    Let us further assume that $\epsilon^{(i)}$ are distributed IID (independently and identically distributed) according to a Gaussian distribution with mean zero and variance $\sigma^2$, i.e. &nbsp; $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$. Thus, we have:
    
    $$\begin{align*}
    p(\epsilon^{(i)}) &= \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(\epsilon^{(i)} - \mu)^2}{2\sigma^2} \right) \\
                      &= \frac{1}{\sqrt{2 \pi \sigma}} \exp \left( -\frac{(\epsilon^{(i)})^2}{2\sigma^2} \right) \\
    \end{align*}$$

    Since $y^{(i)} = \theta^{T}x^{(i)} + \epsilon^{(i)}$, this leads to 
    
    $$\begin{align*}
    p(y^{(i)} \mid x^{(i)}; \theta) = \frac{1}{\sqrt{2 \pi \sigma}} \exp \left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)
    \end{align*}$$

    Considering all samples $\mathbf{X}$, we use the likelyhood function $L(\theta) = L(\theta; X, \vec{y}) = p(\vec{y} \mid X; \theta)$ to denote the probability of targets vector $\vec{y}$. Since $\epsilon^{(i)}$ are distributed independently, we have:
    
    $$\begin{align*}
    L(\theta) &= \prod_{i=1}^n p(y^{(i)} \mid x^{(i)}; \theta) \\
              &= \prod_{i=1}^n \frac{1}{\sqrt{2 \pi \sigma}} \exp \left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2 \sigma^2} \right)
    \end{align*}$$

    Now, we naturally want to increase the probability of $L(\theta)$, which is to choose a proper $\theta$ to _maximize likelyhood_. Instead of maximizing $L(θ)$, we can also maximize any strictly increasing
    function of $L(θ)$. In particular, the derivations will be a bit simpler if we
    instead maximize the log likelihood $\ell(\theta)$:
    
    $$\begin{align*}
    \ell(\theta) &= \log L(\theta) \\
                 &= \log \prod_{i=1}^n \left( \frac{1}{\sqrt{2\pi\sigma}} \exp \left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right) \right) \\
                 &= \sum_{i=1}^n \log \left( \frac{1}{\sqrt{2\pi\sigma}} \exp \left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right) \right) \\
                 &= n \log \left( \frac{1}{\sqrt{2\pi\sigma}} \right) - \frac{1}{2\sigma^2} \sum_{i=1}^n (y^{(i)} - \theta^T x^{(i)})^2
    \end{align*}$$

    Hence, maximizing $\ell(\theta)$ is equivalent to minimizing 
    
    $$\begin{align*}
    \frac{1}{2} \sum_{i=1}^n (y^{(i)} - \theta^T x^{(i)})^2
    \end{align*}$$

## Maximum Likelihood Estimation (MLE)
!!! df "**Definition** (Maximum Likelihood Estimation)"
    Maximum Likelihood Estimation (MLE) is the process of estimating the parameters of a distribution that maximize the likelihood of the observed data belonging to that distribution. [^1]
    [^1]: [_Understanding Maximum Likelihood Estimation_](ihttps://polaris000.medium.com/understanding-maximum-likelihood-estimation-e63dff65e5b1#:~:text=Simply%20put%2C%20when%20we%20perform,with%20the%20principles%20of%20MLE.) by Aniruddha Karajg

## Generalization
!!! df "**Definition** (Underfitting, Overfitting and Regularization)"
    Underfitting is the phenomenon that the model is unable to reduce the training error and to capture the pattern that we are trying to model. _Overfitting_ is the phenomenon of fitting closer to the training data than to the underlying distribution, and techniques for combatting overfitting are often called _regularization_ methods.

!!! df "**Definition** (Training Error, Generalization Error, and Validation Error)"
    The _training error_ $R_{\text{emp}}$ is a statistic calculated on the training dataset, and the _generalization error_ $R$ is an expectation taken with respect to the underlying distribution. Generalization error is what you would see if applying the model to an infinite stream of additional data examples drawn from the same underlying data distribution. Formally the training error is expressed as a **sum**:

    $$
    R_{\text{emp}}[\mathbf{X}, \mathbf{y}, f] = \frac{1}{n} \sum_{i=1}^{n} l(\mathbf{x}^{(i)}, y^{(i)}, f(\mathbf{x}^{(i)})),
    $$

    while the generalization error is expressed as an **integral**:

    $$
    R[p, f] = \mathbb{E}_{(\mathbf{x}, y) \sim P} [l(\mathbf{x}, y, f(\mathbf{x}))] = \iint l(\mathbf{x}, y, f(\mathbf{x})) p(\mathbf{x}, y) \, d\mathbf{x} \, dy.
    $$

    We can never calculate the generalization error $R$ exactly. In practice, we must **estimate** the generalization error by applying our model to an independent test set constituted of a random selection of examples $\mathbf{X}'$ and labels $\mathbf{y}'$ that were withheld from our training set. This consists of applying the same formula that was used for calculating the empirical training error but to a test set $\mathbf{X}', \mathbf{y}'$.

    Error on the holdout data, i.e., validation set, is called the _validation error_.

!!! mt "**Methodology** (Underfitting or Overfitting?)"
    When our training error and validation error are both substantial but there is a little gap between them, then it may be an underfitting. <br>
    If the training error is significantly lower than the validation error, then it may be an overfitting.

!!! st "**Strategy** (Cross-Validation)"
    $K$-fold cross-validation: Splitting the original dataset into non-overlapping subsets. Then model training and validation are executed $K$ times, each time training on subsets and validating on a different subset (the one not used for training in that round). Finally, the training and validation errors are estimated by averaging over the results from the experiments.

!!! nt "**Note** (Fixing Overfitting)"
    A very blunt way to fix overfitting is to reduce the number of parameters (weights). The intuition is: a higher degree polynomial can depict a more complicated model than a lower degree polynomial. So if #weights is large, then the polynomial may fit in with the training dataset too well.

## Weight Decay
!!! df "**Definition** (Weight Decay)"
    Rather than directly manipulating the number of parameters, weight decay, operates by restricting the values that the parameters can take. More commonly called $\ell_2$ regularization outside of deep learning circles when optimized by minibatch stochastic gradient descent, weight decay might be the most widely used technique for regularizing parametric machine learning models.

    - [ ] Weight Decay Interpretation
    - [ ] Lagrange Multiplier Interpretation
    - [ ] Bayes Interpretation

## Analytical Solution of Linear Regression
!!! df "**Definition** (Projection Matrix)"
    The projection matrix of vector $\mathbf{v}$ is the outer product of $\mathbf{v}$ and itself &nbsp; "divides by" their inner product:

    $$
    \frac{\mathbf{v}\mathbf{v}^T}{\mathbf{v}^T\mathbf{v}}
    $$

    The projection matrix of matrix $\mathbf{X}$ is also the outer product of $\mathbf{X}$ and itself &nbsp; "divides by" their inner product. Here we multiply the inverse of $\mathbf{X}^T\mathbf{X}$:

    $$
    \mathbf{X}{(\mathbf{X}^T\mathbf{X})}^{-1}\mathbf{X}^T 
    $$

!!! nt "**Note**"
    We put ${(\mathbf{X}^T\mathbf{X})}^{-1}$ in the middle because only $\mathbf{X}{(\mathbf{X}^T\mathbf{X})}^{-1}\mathbf{X}^T$: $(m\times n)\times((n\times m)\times(m\times n))\times(n\times m)$ makes sense when we do calculation.
      
!!! tm "**Theorem** (Computing Projection)"
    The projection of vector $\mathbf{b}$ on the vector $\mathbf{v}$ is

    $$
    \text{projection matrix}(v) \cdot \mathbf{b}
    $$

    ??? pf "**Proof**"
        Note that $\mathbf{v}^T\mathbf{v}=\|\mathbf{v}\|$. Thus we have:

        $$\begin{align}
        \text{projection matrix}(v) \cdot \mathbf{b}
        &= \frac{\mathbf{v}\mathbf{v}^T}{\mathbf{v}^T\mathbf{v}}\cdot\mathbf{b} \\
        &= \frac{\mathbf{v}}{\mathbf{\|\mathbf{v}\|}}\cdot\frac{\mathbf{v}^T}{\mathbf{\|\mathbf{v}\|}}\cdot\mathbf{b} \\
        &= \tilde{v}\cdot(\tilde{v})^T\cdot\mathbf{b} \\
        &= \tilde{v}\cdot((\tilde{v})^T\mathbf{b})
        \end{align}$$

        Note that $(\tilde{v})^T\mathbf{b}$ is the length of the projection of $\mathbf{b}$ on $\mathbf{v}$. Thus, $\tilde{v}\cdot((\tilde{v})^T\mathbf{b})$ is exactly the projection of $\mathbf{b}$ on $\mathbf{v}$.