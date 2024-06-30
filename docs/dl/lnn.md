# Linear Regression

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
