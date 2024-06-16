# Linear Regression

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
