# Matrix Multiplication
## Some Clarifications
Just some minor problems, but I thought I'd put it clear here.<br>
When we talk about matrices, $a_{41}$ is the first element in the forth row, and an $m \times n$ matrix is a matrix of $m$ rows and $n$ columns.
## Different Ways to Do Multiplication
Apart from the inner product method (or we can call it the "row time column" method) in the first chapter, there are some other ways. Consider:

$$
A=
\begin{bmatrix}
1&3\\2&4\\3&0
\end{bmatrix}
\begin{bmatrix}
3&2\\3&1
\end{bmatrix}
$$

!!!df Column

    $$
    \begin{aligned}
    A&=
    \begin{bmatrix}
    \begin{bmatrix}
    1&3\\2&4\\3&0
    \end{bmatrix}
    \begin{bmatrix}
    3\\3
    \end{bmatrix}
    &\begin{bmatrix}
    1&3\\2&4\\3&0
    \end{bmatrix}
    \begin{bmatrix}
    2\\1
    \end{bmatrix}
    \end{bmatrix}\\
    &=
    \begin{bmatrix}
    10&5\\18&8\\9&6
    \end{bmatrix}
    \end{aligned}
    $$

    Which implicates that each column of the product is a combination of the columns of the matrix $\begin{bmatrix}1&3\\2&4\\3&0\end{bmatrix}$.

!!!df Row

    $$
    \begin{aligned}
    A&=
    \begin{bmatrix}
    \begin{bmatrix}
    1&3
    \end{bmatrix}
    \begin{bmatrix}
    3&2\\3&1
    \end{bmatrix}
    \\\begin{bmatrix}
    2&4
    \end{bmatrix}
    \begin{bmatrix}
    3&2\\3&1
    \end{bmatrix}
    \\\begin{bmatrix}
    3&0
    \end{bmatrix}
    \begin{bmatrix}
    3&2\\3&1
    \end{bmatrix}
    \end{bmatrix}\\
    &=
    \begin{bmatrix}
    10&5\\18&8\\9&6
    \end{bmatrix}
    \end{aligned}
    $$

!!!df Column times Row

    $$
    \begin{aligned}
    A&=
    \begin{bmatrix}
    1\\2\\3
    \end{bmatrix}
    \begin{bmatrix}
    3&2
    \end{bmatrix}
    +\begin{bmatrix}
    3\\4\\0
    \end{bmatrix}
    \begin{bmatrix}
    3&0
    \end{bmatrix}\\
    &=
    \begin{bmatrix}
    10&5\\18&8\\9&6
    \end{bmatrix}
    \end{aligned}
    $$

!!!df Chunk
    For this one, suppose two big matrices $A$ and $B$ that can be divided in to four chunks:

    $$
    \begin{cases}
    A=\begin{bmatrix}
    A_1&A_2\\A_3&A_4
    \end{bmatrix}\\
    B=\begin{bmatrix}
    B_1&B_2\\B_3&B_4
    \end{bmatrix}
    \end{cases}
    $$

    where $A_1$, $A_2$, $A_3$, $A_4$, $B_1$, $B_2$, $B_3$, $B_4$ are all matrices. Suppose they have the right dimensions.
    Their product is:

    $$
    AB=\begin{bmatrix}
    A_1B_1+A_2B_3&A_1B_2+A_2B_4\\
    A_3B_1+A_4B_3&A_3B_2+A_4B_4
    \end{bmatrix}
    $$