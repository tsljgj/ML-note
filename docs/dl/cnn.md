# Convolutional Neural Networks (CNN)

## Convolutions
!!! df "**Definition** (Convolutions)"
    In mathematics, the _convolution_ between two functions $f,g: \mathbb{R}^d \rightarrow \mathbb{R}$ is defined as 

    $$\begin{align*}
    (f * g)(x) = \int f(t)g(x-t)dt
    \end{align*}$$

    We measure the overlap between $f$ and $g$ when one function is "flipped" and shifted by $x$.

??? eg "**Example** (Dice - Not a Good Example)"
    Assume there are two 6-face dices. We want to know the probability of the sum of two dices equals to 4. <br>
    Define $f(x) = \text{probability of getting x on dice 1}$, $g(x) = \text{probability of getting x on dice 2}$. The probability of getting a sum of 4 is:
    
    $$\begin{align*}
    (f * g)(4) = \sum^3_{m=1}f(4-m)g(m) = f(1)g(3) + f(2)g(2) + f(3)g(1)
    \end{align*}$$

    We can view this as first flipped the function $g$, and then shift $1,2,3,4$ position and calculate the overlapping area.

    <figure markdown="span">
    ![Image title](../../graphs/dl/cnn/cnn_dice.svg){ width="400" }
    </figure>

!!! im "**Important Note** (Convolutions are Impact Sum)"
    

    
     
    
    
 

