# Elimination Matrix

## Elimination
We shall begin this chapter with a systematic way to solve linear equations with n variables. Consider the following set of equations where n = 3:

$$
\begin{cases}
&x &+ &2&y &+ &z &= &2\\
3&x &+ &8&y &+ &z &= &12\\
&&&4&y &+ &z &= &2
\end{cases}
$$

Which would correspond to the coefficient matrix $A$:

$$
A=
\begin{bmatrix}
1&2&1\\3&8&1\\0&4&1
\end{bmatrix}
$$

A way to systematically solve the equation is called elimination. We begin by eliminating $x$ in the second equation, and from then on eliminate $x$ & $y$ in the third equation.

???pf "**Intermediate Steps**"
    $$
    \begin{aligned}
    &\begin{cases}
    &x &+ &2&y &+ &z &= &2\\
    3&x &+ &8&y &+ &z &= &12\\
    &&&4&y &+ &z &= &2
    \end{cases}\\
    \Rightarrow
    &\begin{cases}
    &x &+ &2&y &+ &&z &= &2\\
    &&&2&y &- &2&z &= &6\\
    &&&4&y &+ &&z &= &2
    \end{cases}\\
    \Rightarrow
    &\begin{cases}
    &x &+ &2&y &+ &&z &= &2\\
    &&&2&y &- &2&z &= &6\\
    &&&&&&5&z &= &-10
    \end{cases}
    \end{aligned}
    $$

In this case, we would end up with a matrix like this:

$$
U=
\begin{bmatrix}
1&2&1\\0&2&-2\\0&0&5
\end{bmatrix}
$$

Such a matrix is called an **Upper Triangular Matrix**, denoted by $U$.<br>
If we want to solve the equation, we can add the right sides of the equations to the matrix as a column on the right side of the matrix to get an **Augmented Matrix**.From then on we can go over the same process of elimination. But this is not the point for now.

## Elimination Matrix
If we look back on what we just did, we were essentially doing row operations: subtracting three times the first row from the second row, etc. Now if you remember what we just discovered in the first chapter about row operations, you might think that we might be able to multiply something by the matrix $A$ to get the same effect, namely:

$$
EA=U
$$

where $E$ is a matrix we call the **Elementary Matrix**. Since $U$ and $A$ are three by three matrices, $E$ should be as well.<br>
Think about it step by step: first let us leave the third row and get the $x$ in the second row eliminated. The operation here is to subtract three times the first row from the second row, and leave everything else unchanged. Since the first row and third row are not changed, the first and third row of $E$ should be $\begin{bmatrix}1&0&0\end{bmatrix}$ and $\begin{bmatrix}0&0&1\end{bmatrix}$ respectively. Since we want to subtract three times the first row from the second row of $A$, the second row of $E$ would be: $\begin{bmatrix}-3&1&0\end{bmatrix}$. The resulting matrix $E$ is then:

$$
E_{21}=
\begin{bmatrix}
1&0&0\\-3&1&0\\0&0&1
\end{bmatrix}
$$

???pf "**Identity Matrix**"
    As you may have discovered, we can also multiply something by the matrix $A$ for it to remain unchanged. Such a matrix is called the **Identity Matrix** $I$. It takes the following form:
    $$
    I=
    \begin{bmatrix}
    1&0&0&\cdots&0\\
    0&1&0&\cdots&0\\
    0&0&1&\cdots&0\\
    \vdots&\vdots&\vdots&\ddots&\vdots\\
    0&0&0&\cdots&1
    \end{bmatrix}
    $$
    And it satisfies:
    $$
    \forall A, AI = IA = A
    $$
    given that $A$ is a square matrix.

Through a similar process, we can get the elementary matrix for the next step:

$$
E_{32} = 
\begin{bmatrix}
1&0&0\\
0&1&0\\
0&-2&1
\end{bmatrix}
$$

Such that

$$
U = E_{32} (E_{21}A)
$$

A very important fact is that matrix multiplication follows the ***Assiciative Law***, and we can rewrite the equation as:

$$
\begin{aligned}
U &= (E_{32}E_{21}A)\\
&= E_{31}A
\end{aligned}
$$

But it's also important to know that switching orders is not valid in most matrix multiplications:

$$
AB \not = BA
$$

in most cases.

???pf "**Permutation Matrix**"
    Of course, you can also switch the rows of a matrix by multiplication, for example:
    $$
    \begin{bmatrix}
    0&1\\
    1&0
    \end{bmatrix}
    \begin{bmatrix}
    a&b\\
    c&d
    \end{bmatrix}
    =\begin{bmatrix}
    c&d\\
    a&b
    \end{bmatrix}
    $$
    Notably,
    $$
    \begin{bmatrix}
    a&b\\
    c&d
    \end{bmatrix}
    \begin{bmatrix}
    0&1\\
    1&0
    \end{bmatrix}
    =\begin{bmatrix}
    b&a\\
    d&c
    \end{bmatrix}
    $$