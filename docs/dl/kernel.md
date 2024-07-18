# Kernel Methods

!!! df "**Definition** (Feature Map)"
    In a non-linear model, we call the original input $x$ the input **attributes**. The **feature** variables now become the output of a function $\phi : {\mathbb{R}}^d \rightarrow {\mathbb{R}}^p$, where $d$ is the #dimension of $x$, and $p$ is the #dimension of the feature. We call $\phi$ a _Feature Map_.

??? eg "**Example** (Example of Feature Map)"
    We considered the problem of predicting the price of a house (denoted by \( y \)) from the living area of the house (denoted by \( x \)). The price \( y \) can be more accurately represented as a **non-linear** function of \( x \).

    Consider fitting cubic functions \( y = \theta_3 x^3 + \theta_2 x^2 + \theta_1 x + \theta_0 \). We can view the cubic function as a linear function over a different set of feature variables. Concretely, let the function \( \phi : \mathbb{R} \to \mathbb{R}^4 \) be defined as

    \[
    \phi(x) = \begin{bmatrix}
    1 \\
    x \\
    x^2 \\
    x^3 \\
    \end{bmatrix} \in \mathbb{R}^4.
    \]

    Let \( \theta \in \mathbb{R}^4 \) be the vector containing \( \theta_0, \theta_1, \theta_2, \theta_3 \) as entries. Then we can rewrite the cubic function in \( x \) as:

    \[
    \theta_3 x^3 + \theta_2 x^2 + \theta_1 x + \theta_0 = \theta^T \phi(x)
    \]

    Thus, a cubic function of the variable \( x \) can be viewed as a linear function over the variables \( \phi(x) \).

!!! im "**Important Note** (LMS (Least Mean Squares) with Features)"
    We will derive the gradient descent algorithm for fitting the model \( \theta^T \phi(x) \). First recall that for ordinary least square problem where we were to fit \( \theta^T x \), the batch gradient descent update is:

    $$\begin{align*}
    \theta &:= \theta + \alpha \sum_{i=1}^{n} \left( y^{(i)} - h_{\theta}(x^{(i)}) \right) x^{(i)} \\
           &:= \theta + \alpha \sum_{i=1}^{n} \left( y^{(i)} - \theta^T x^{(i)} \right) x^{(i)}.
    \end{align*}$$

    Let \( \phi : \mathbb{R}^d \to \mathbb{R}^p \) be a feature map that maps attribute \( x \) (in \( \mathbb{R}^d \)) to the features \( \phi(x) \) in \( \mathbb{R}^p \). (In the previous example, we have \( d = 1 \) and \( p = 4 \).) Now our goal is to fit the function \( \theta^T \phi(x) \), with \( \theta \) being a vector in \( \mathbb{R}^p \) instead of \( \mathbb{R}^d \). We can replace all the occurrences of \( x^{(i)} \) in the algorithm above by \( \phi(x^{(i)}) \) to obtain the new update:

    \[
    \theta := \theta + \alpha \sum_{i=1}^{n} \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \phi(x^{(i)}).
    \]

    Similarly, the corresponding stochastic gradient descent update rule is

    \[
    \theta := \theta + \alpha \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \phi(x^{(i)}).
    \]

The gradient descent update, or stochastic gradient update above becomes computationally expensive when the features $\phi(x)$ is high-dimensional. It may appear at first that such runtime per update and memory usage are inevitable, because the vector $\phi$ itself is of the same dimension as $\phi(x)$, and we may need to update every entry of $\phi$ and store it. However, we will introduce the
kernel trick with which we will not need to store $\phi$ explicitly, and the runtime can be significantly improved.<br>

For simplicity, we assume the initialize the value \( \theta = 0 \), and we focus on the iterative update. 

!!! tm "**Statement**"
    \( \theta \) can be represented as a linear combination of the vectors \( \phi(x^{(1)}), \ldots, \phi(x^{(n)}) \). 
    
    ??? pf "**Proof**"
        We proceed by induction. The base case is: \( \theta = 0 = \sum_{i=1}^{n} 0 \cdot \phi(x^{(i)}) \). Assume at some point, \( \theta \) can be represented as

        \[
        \theta = \sum_{i=1}^{n} \beta_i \phi(x^{(i)})
        \]

        for some \( \beta_1, \ldots, \beta_n \in \mathbb{R} \). Then we claim that in the next round, \( \theta \) is still a linear combination of \( \phi(x^{(1)}), \ldots, \phi(x^{(n)}) \) because 
        
        $$\begin{align*}
        \theta :&= \theta + \alpha \sum_{i=1}^{n} \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \phi(x^{(i)}) \\
               &= \sum_{i=1}^{n} \beta_i \phi(x^{(i)}) + \alpha \sum_{i=1}^{n} \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \phi(x^{(i)}) \\
               &= \sum_{i=1}^{n} \left( \beta_i + \alpha \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \right) \phi(x^{(i)})
        \end{align*}$$

!!! im "**Important Note** (Transforming GD into Updating \( \beta \))"
    Our general strategy is to implicitly represent the \( p \)-dimensional vector \( \theta \) by a set of coefficients \( \beta_1, \ldots, \beta_n \). By previous calculation, we derive the update rule of the coefficients \( \beta_1, \ldots, \beta_n \):

    \[
    \beta_i := \beta_i + \alpha \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right)
    \]

    Here we still have the old \( \theta \) on the RHS of the equation. Replacing \( \theta = \sum_{j=1}^{n} \beta_j \phi(x^{(j)}) \) gives

    \[
    \forall i \in \{1, \ldots, n\}, \beta_i := \beta_i + \alpha \left( y^{(i)} - \sum_{j=1}^{n} \beta_j \phi(x^{(j)})^T \phi(x^{(i)}) \right)
    \]

    We often rewrite \( \phi(x^{(j)})^T \phi(x^{(i)}) \) as \( \langle \phi(x^{(j)}), \phi(x^{(i)}) \rangle \) to emphasize that it's the inner product of the two feature vectors. Viewing \( \beta_i \)'s as the new representation of \( \theta \), we have successfully translated the batch gradient descent algorithm into an algorithm that updates the value of \( \beta \) iteratively. It may appear that at every iteration, we still need to compute the values of \( \langle \phi(x^{(j)}), \phi(x^{(i)}) \rangle \) for all pairs of \( i, j \), each of which may take roughly \( O(p) \) operation. However, two important properties come to rescue:

    1. We can pre-compute the pairwise inner products \( \langle \phi(x^{(j)}), \phi(x^{(i)}) \rangle \) for all pairs of \( i, j \) before the loop starts.

    2. For many feature map \( \phi \), computing \( \langle \phi(x^{(j)}), \phi(x^{(i)}) \rangle \) can be efficient.

!!! im "**Important Note** (Computing \( \langle \phi(x^{(j)}), \phi(x^{(i)}) \rangle \))"
    In previous example, we can compute \( \langle \phi(x^{(j)}), \phi(x^{(i)}) \rangle \) without computing \( \phi(x^{(i)}) \) explicitly:

    $$\begin{align*}
    \langle \phi(x), \phi(z) \rangle &= 1 + \sum_{i=1}^{d} x_i z_i + \sum_{i,j \in \{1, \ldots, d\}} x_i x_j z_i z_j + \sum_{i,j,k \in \{1, \ldots, d\}} x_i x_j x_k z_i z_j z_k \\
    &= 1 + \sum_{i=1}^{d} x_i z_i + \left( \sum_{i=1}^{d} x_i z_i \right)^2 + \left( \sum_{i=1}^{d} x_i z_i \right)^3 \\
    &= 1 + \langle x, z \rangle + \langle x, z \rangle^2 + \langle x, z \rangle^3
    \end{align*}$$

    Therefore, to compute \( \langle \phi(x), \phi(z) \rangle \), we can first compute \( \langle x, z \rangle \) with \( O(d) \) time and then take another constant number of operations to compute \( 1 + \langle x, z \rangle + \langle x, z \rangle^2 + \langle x, z \rangle^3 \).

!!! df "**Definition** (Kernel)"
    The inner products between the features \( \langle \phi(x), \phi(z) \rangle \) are essential throughout our calculation. Thus, we define the _Kernel_ corresponding to the feature map \( \phi \) as a function that maps \( \mathcal{X} \times \mathcal{X} \to \mathbb{R} \) satisfying: 

    \[
    K(x, z) \triangleq \langle \phi(x), \phi(z) \rangle
    \]

    Note that \( \mathcal{X} \) is the space of the input $x$. In our example, \( \mathcal{X} = \mathbb{R}^d \).

!!! df "**Model** (Final Algorithm with Kernel)"
    We write down the final algorithm as follows:

    1. Compute all the values \( K(x^{(i)}, x^{(j)}) \triangleq \langle \phi(x^{(i)}), \phi(x^{(j)}) \rangle \) for all \( i, j \in \{1, \ldots, n\} \). Set \( \beta := 0 \).

    2. **Loop:**

    \[
    \forall i \in \{1, \ldots, n\}, \beta_i := \beta_i + \alpha \left( y^{(i)} - \sum_{j=1}^{n} \beta_j K(x^{(i)}, x^{(j)}) \right)
    \]

    Or in vector notation, letting \( K \) be the \( n \times n \) matrix with \( K_{ij} = K(x^{(i)}, x^{(j)}) \), we have

    \[
    \beta := \beta + \alpha (\vec{y} - K \beta)
    \]
    
    With the algorithm above, we can update the representation \( \beta \) of the vector \( \theta \) efficiently with \( O(n) \) time per update. 

!!! im "**Important Note** (Computing the Prediction with Kernel)"
    The knowledge of the representation \( \beta \) suffices to compute the prediction \( \theta^T \phi(x) \):

    \[
    \theta^T \phi(x) = \sum_{i=1}^{n} \beta_i \phi(x^{(i)})^T \phi(x) = \sum_{i=1}^{n} \beta_i K(x^{(i)}, x)
    \]

    Fundamentally, all we need to know about the feature map \( \phi(\cdot) \) is encapsulated in the corresponding kernel function \( K(\cdot, \cdot) \).

!!! nt "**Note** (Limitation of Kernel)"
    Note that to make a prediction, we need to store all training dataset, which can be an obstacle for Kernel method.

- [ ] Property of Kernel
- [ ] How to define a valid Kernel

## Support Vector Machine

- [ ] Support Vector Machine
    