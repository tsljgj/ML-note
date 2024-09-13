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
