# Linear Regression: Softmax
In addition to the house price prediction problem, we are interested in another kind of linear regression problem: classification. Instead of asking "how much," we now ask "which."

## Classification Problem
!!! df "**Definition** (Classification Problem)"
    There are two kinds of _classification_ problem:<br>
    &nbsp;&nbsp;&nbsp;&nbsp;1. Hard assignments of examples to categories<br>
    &nbsp;&nbsp;&nbsp;&nbsp;2. Assess the probaility that each category applies

??? eg "**Example** (Classification Problem Example)"
    Consider a $2 \times 2$ image. Let $x_1, x_2, x_3, x_4$ denote the scalar value of each pixel. These are the four features in our model. Furthermore, assume that each image belongs to one of the categories "cat," "chicken," or "dog." We want to determine the class to which a given image belongs.

!!! df "**Definition** (One-hot Encoding)" 
    For a sample in an $n$-category classification problem, the _hot-encoding_ of that sample is a vector with $n$ components, and the only component corresponding to the sample's category is set to 1 and all other components are set to 0.

??? eg "**Example** (One-hot Encoding Example)"
    In the previous example, $n = 3$, so "cat" can be encoded as $(1, 0, 0)$, "chicken" ... $(0, 1, 0)$, "dog" ... $(0, 0, 1)$.

!!! nt "**Note** (Ordered-Category Encoding)"
    If categories had some natural ordering among them, we can encode them in a much more intuitive way. For example, say we want to predict $\{\text{baby, toddler, adolescent}\}$, then it might make sense to cast this as an ordinal regression problem and keep the labels in this format: $\text{label} \in \{1, 2, 3\}$

!!! df "**Model** (Linear Model for Classification)"
    Consider a network with one output layer and one input layer. For classification problems, #nodes in the output layer $=$ #category.
    Let $\mathbf{o}$ denotes the output (a vector) of neural network, we have:
    
    $$\mathbf{o} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

    <figure markdown="span">![Image title](https://d2l.ai/_images/softmaxreg.svg){ width="500" }<figcaption>Linear Model for Classification Problem </figcaption></figure>
    The above model represents a $2 \times 2$ image and 3-category classification problem. The output $\mathbf{o}$ is thus calculated by:
    
    $$
    \begin{align*}
    o_1 &= w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 \\
    o_2 &= w_5x_1 + w_6x_2 + w_7x_3 + w_8x_4 \\
    o_3 &= w_9x_1 + w_{10}x_2 + w_{11}x_3 + w_{12}x_4
    \end{align*}
    $$

At this point, we can, assuming a suitable loss function, try to minimize the difference between $\mathbf{o}$ and the hot-encoding of the sample. In fact, this works surprisingly well. However, we need to consider two drawbacks of this method:<br>
&nbsp;&nbsp;&nbsp;&nbsp;1. $\sum o_i \neq 1$<br>
&nbsp;&nbsp;&nbsp;&nbsp;2. $o_i$ may be negative<br>
To address these issues, we use _softmax_.

## Softmax Function
!!! df "**Definition** (Softmax Function)"
    $$
    \hat{\mathbf{y}} = \text{softmax}(\mathbf{o}) \; \text{ where } \; \hat{y}_i = \frac{\text{exp}(o_i)}{\sum_{j}\text{exp}(o_j)}.
    $$

!!! nt "**Note**"
    By softmax definition, the largest coordinate of $\mathbf{o}$ corresponds to the most likely class. We do not even need to compute softmax to determine which class has the highest possibility.

!!! im "**Important Note** (Vectorization)"
    To improve computational efficiency, we vectorize calculations in minibatches of data. Assume that we are given a minibatch $\mathbf{X} \in \mathbb{R}^{n\times d}$ ($n$ samples with $d$ dimensions). Moreover, assume we have $q$ categories. Then we have $\mathbf{W} \in \mathbb{R}^{d\times q}$ and bias $\mathbf{b} \in \mathbb{R}^{1\times d}$. Finally, we have:
     
    $$
    \begin{align}
    \mathbf{O} &= \mathbf{X}\mathbf{W} + \mathbf{b} \\
    \hat{\mathbf{Y}} &= \text{softmax}(\mathbf{O})
    \end{align}
    $$
    
    This accelerates the dominant operation into a matrix–matrix product.

!!! rm "**Remark** (Viewing Softmax Output as Likelihood)"
    The softmax function gives us a vector $\hat{\mathbf{y}}$. We can interpret it as the estimated conditional probabilities of each category, given any input $\mathbf{x}$. e.g. $\hat{y}_1=P(y=(1,0,0) \mid \mathbf{x})$. Thus we have 
    
    $$
    P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
    $$  

## Maximum Likelihood Estimation (MLE)
!!! df "**Definition** (Maximum Likelihood Estimation)"
    Maximum Likelihood Estimation (MLE) is the process of estimating the parameters of a distribution that maximize the likelihood of the observed data belonging to that distribution. [^1]
    [^1]: [_Understanding Maximum Likelihood Estimation_](ihttps://polaris000.medium.com/understanding-maximum-likelihood-estimation-e63dff65e5b1#:~:text=Simply%20put%2C%20when%20we%20perform,with%20the%20principles%20of%20MLE.) by Aniruddha Karajg