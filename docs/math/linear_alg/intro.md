# Starting Point of Linear Algebra

## Geometries of Equations
Essentially, Linear Algebra starts with solving n equations with m variables. To make things easy, we make n equal to m. For example:
$$
2x - y = 0\\
-x + 2y = 3
$$
In the language of Linear Algebra would be:
$$
\begin{bmatrix}
2&-1\\
-1&2
\end{bmatrix}
\begin{bmatrix}
x\\
y
\end{bmatrix}=
\begin{bmatrix}
0\\
3
\end{bmatrix}
$$
Where each of the numbers in the matrix represent a coefficient on the left hand side, and $x$ & $y$ represent the variables.

## Row Picture and Column Picture
If you carefully think about it, there are two ways you can represent the equation with a picture:
First, you can think about it as two lines intersecting each other, with $2x - y = 0$ being the first one, and $-x + 2y = 3$ being the second one:
<figure markdown="span">
![Row Image](./assets/images/row_picture.png "Row Image")
<figcaption>Generated Using Geogebra</figcaption>
</figure>
Or you can think about it as a Vector Addition problem:
$$
x\begin{bmatrix}2\\-1\end{bmatrix} + y
\begin{bmatrix}-1\\2\end{bmatrix} = 
\begin{bmatrix}0\\3\end{bmatrix}
$$