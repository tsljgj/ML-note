# Inverses

## Inverses
!!!df Definition
    An Inverse of the matrix $A$ is $A^{-1}$ such that $A^{-1}A=I$ where $I$ is the Identity Matrix. Note that $AA^{-1}=I$ also holds
    Inverses do not exist when there is a non-trivial vector $\vec x$ such that $A\vec x=\vec0$

    ???pf "proof"
        Suppose $A^{-1}$ exists in this case.<br>
        Then $A^{-1}A\vec x=A^{-1}\vec 0$<br>
        $\Rightarrow I\vec x = \vec 0$
        $\Rightarrow \vec x = \vec 0$

## Finding Inverses
!!!df "Gause-Jordan Method"
    Consider a matrix $A$:

    $$
    A=
    \begin{bmatrix}
    1&3\\
    2&7
    \end{bmatrix}
    $$

    and we want to find $A^{-1}$ such that $A^{-1}A=I$.<br>
    Suppose that $A^{-1}=\begin{bmatrix}a&b\\c&d\end{bmatrix}$, then it is actually two equation sets:

    $$
    \begin{cases}
    \begin{bmatrix}a&b\end{bmatrix}
    \begin{bmatrix}1&3\\2&7\end{bmatrix}
    =\begin{bmatrix}1&0\end{bmatrix}\\
    \begin{bmatrix}c&d\end{bmatrix}
    \begin{bmatrix}1&3\\2&7\end{bmatrix}
    =\begin{bmatrix}0&1\end{bmatrix}
    \end{cases}
    $$

    One way to find $A^{-1}$ is the **Gause-Jordan** method, and it is achieved first by adding the Identity Matrix to the right of $A$ to form an Augmented Matrix $\begin{bmatrix}1&3&1&0\\2&7&0&1\end{bmatrix}$. Then we perform elimination until the left side of the matrix is the Identity Matrix: $\begin{bmatrix}1&0&7&-3\\0&1&-2&1\end{bmatrix}$, and the right side of the matrix, $\begin{bmatrix}7&-3\\-2&1\end{bmatrix}$ is $A^{-1}$. You can verify for yourself.
    ???pf "Proof that this is valid"
        Essentially we can rewrite the elimination process with a Elimination matrix, $E$: what we are doing is essentially $E\begin{bmatrix}A&I\end{bmatrix}$. Since the left side of the resulting product is $I$, we know that $EA=I$, and so $E=A^{-1}$. Since the right side of the original augmented matrix is $I$, the resulting right side is $EI=E=A^{-1}$.