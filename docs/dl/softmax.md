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

    To maximize $P(\mathbf{Y} \mid \mathbf{X})$, we minimize $-\log{P(\mathbf{y} \mid \mathbf{x})}$: 
    
    $$\begin{align*}
    -\log{P(\mathbf{Y} \mid \mathbf{X})} = \sum_{i=1}^n{-\log{P(\mathbf{y} \mid \mathbf{x})}} = \sum_{i=1}^n{l({\mathbf{y}}^{(i)}, {\hat{\mathbf{y}}}^{(i)})}
    \end{align*}$$

    where 
    
    $$\begin{align*}
    l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum^q_{j=1}y_j\log{\hat{y_j}}
    \end{align*}$$

    $-\sum^q_{j=1}y_j\log{\hat{y_j}}$ is also called _Cross-Entrophy_, which will be introduced later in information theory.

## Brief Information Theory
!!! df "**Definition** (Self-Information)"
    Shannon defined the _self-information_ $I(X)$ of event $X$ which has a probability of $p$ as 
    
    $$\begin{align*}
    I(X) = -\log_2{p}
    \end{align*}$$

    as the _bits_ of information we have received for event $X$. For example, 
    
    $$I(\text{"0010"}) = -\log_2{(p(\text{"0010"}))} = -\log_2{(\frac{1}{2^4})} = 4 \text{bits}$$

!!! nt "**Note** (Log Base 2)"
    When discussing information theory, the default base of $\log$ is $2$ instead of $e$ as $2$ nicely corresponds to the unit "bit".

!!! df "**Definition** (Entrophy)"
    For any random variable \( X \) that follows a probability distribution \( P \) with a probability density function (p.d.f.) or a probability mass function (p.m.f.) \( p(x) \), we measure the expected amount of information through entropy $H(X)$ (or Shannon entropy):

    $$
    H(X) = -E_{x \sim P}[\log p(x)].
    $$

    To be specific, if \( X \) is discrete,

    $$
    H(X) = -\sum_i p_i \log p_i, \text{ where } p_i = P(X_i).
    $$

    Otherwise, if \( X \) is continuous, we also refer to entropy as differential entropy

    $$
    H(X) = -\int_x p(x) \log{p(x)} \, dx.
    $$

!!! im "**Important Note** (Why Expectation?)"
    Why Expectation? Suppose there's a soccer match between China and Brazil. The probability that China wins is $0.001$ and Brazil $0.999$. If the news says Brazil wins, then there's hardly any information as this is hardly surprising. If China wins, then this is very surprising (abnormal) and contains more information. However, though China winning has more information, it does not mean that the whole system has more information. To estimate the entrophy of the entire system, we thus want to use expectation, which adds up self-information times possibility of events.

!!! nt "**Note** (Why Log?)"
    Why Log? - We want the entropy formula to be additive over independent random variables.
    
!!! nt "**Note** (Why Negative?)"
    Why Negative? - More frequent events should contain less information than less common events, since we often gain more information from an unusual case than from an ordinary one. $\log$ is monotonically increasing with the probabilities, and indeed negative for all values in $[0,1]$. Hence, we add a negative sign in front of function to construct a monotonically decreasing relationship between the probability of events and their entropy, which will ideally be always positive.

How can we compare the difference between two distribution? A naive way may be to calculate their entrophy difference. However, this does not make any sense as two distinct distribution may have the same entrophy. A more proper way is to take one distribution as the standard, and measure "surprises" when the other distribution "see" the standard. e.g. in people's mind, China wins Brazil has $0.001$ possibility, but if China actually wins, people will be very surprised.

!!! df "**Definition** (Kullback-Leibler Divergence)"
    Given a random variable \( X \) that follows the probability distribution \( P \) with a p.d.f. or a p.m.f. \( p(x) \), and we estimate \( P \) by another probability distribution \( Q \) with a p.d.f. or a p.m.f. \( q(x) \). Then the Kullback-Leibler (KL) divergence (or relative entropy) between \( P \) and \( Q \) is
    
    $$\begin{align*}
    D_{KL}(P \parallel Q) &=\sum_{i=1}p_i\cdot(f_Q(q_i)-f_P(p_i)) \\
                          &=\sum_{i=1}p_i\cdot(-\log{q_i}-(-\log{p_i})) \\ 
                          &= E_{x \sim P} \left[ \log \frac{p(x)}{q(x)} \right]
    \end{align*}$$
    
    where $f_Q(q_i)-f_P(p_i)$ is the entrophy difference between the ith event.

!!! df "**Definition** (Cross-Entrophy)"
    Note that 
    
    $$\begin{align*}
    D_{KL}(P \parallel Q) &=\sum_{i=1}{p_i\cdot(f_Q(q_i)-f_P(p_i))} \\
                          &=\sum_{i=1}{p_i\cdot(-\log{q_i}-(-\log{p_i}))} \\ 
                          &=\sum_{i=1}{p_i\cdot (-\log q_i)} - \sum_{i=1}{p_i\cdot (-\log p_i)} \\
                          &=\sum_{i=1}{p_i\cdot (-\log q_i)} - \text{entrophy of distribution P}
    \end{align*}$$

    by Gibbs' inequality, we know that 

    \[
    - \sum_{i=1}^{n} p_i \log p_i \leq - \sum_{i=1}^{n} p_i \log q_i
    \]

    Thus, to minimize KL divergence, it suffices to minimize $\sum_{i=1}{p_i\cdot (-\log q_i)}$. This term is called _Cross-Entrophy_. Formally, for a random variable \( X \), we can measure the divergence between the estimating distribution \( Q \) and the true distribution \( P \) via cross-entropy,

    $$
    CE(P, Q) = -E_{x \sim P}[\log(q(x))].
    $$

    By using properties of entropy discussed above, we can also interpret it as the summation of the entropy \( H(P) \) and the KL divergence between \( P \) and \( Q \), i.e.,

    $$
    CE(P, Q) = H(P) + D_{KL}(P \parallel Q)
    $$    