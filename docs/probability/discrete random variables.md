# Discrete Random Variables
!!! df "**Definition** (Discrete Random Variables)"
    A discrete random variable $X$ on the probability space $(\Omega, \mathcal{F}, \mathbb{P})$ is defined to be a mapping $X: \Omega \rightarrow \mathbb{R}$ such that 
    
    $$
    \begin{align*}
        &\text{the image } X(\Omega) \text{ is a countable subset of } \mathbb{R} \tag{2.1} \\
        &\{\omega \in \Omega : X(\omega) = x\} \in \mathcal{F} \text{ for } x \in \mathbb{R} \tag{2.2} 
    \end{align*}
    $$

!!! nt "**Note** (Interpretation of (2.2))"
    An intuitive Interpretation of (2.2) is: the set of all samples $\omega$ that make $X(\omega) = x$ must be in event space $\mathcal{F}$.

!!! df "**Definition** (Probability Mass Function (pmf))"
    The probability mass function (pmf) of the discrete random variable $X$ is the function \( p_X : \mathbb{R} \to [0, 1] \) defined by 
    
    $$\begin{align*}
    p_X(x) = \mathbb{P}(X = x). \tag{2.4}
    \end{align*}$$

    Thus, \( p_X(x) \) is the probability that the mapping \( X \) takes the value \( x \). Note that \( \text{Im } X \) is countable for any discrete random variable \( X \), and

    \[
    \begin{align*}
        p_X(x) &= 0 \quad \text{if} \quad x \notin \text{Im } X, \tag{2.5} \\
        \sum_{x \in \text{Im } X} p_X(x) &= \mathbb{P} \left( \bigcup_{x \in \text{Im } X} \left\{ \omega \in \Omega : X(\omega) = x \right\} \right) \\
        &= \mathbb{P}(\Omega) = 1. \tag{1.14}
    \end{align*}
    \]

    Because only countable $x$ has non-zero probability, equation (2.6) equivalent to 

    \[
    \begin{align*}
        \sum_{x \in \mathbb{R}} p_X(x) &= 1, \tag{2.6}
    \end{align*}
    \]

Condition (2.6) essentially characterizes mass functions of discrete random variables in the sense of the following theorem.

!!! tm "**Theorem** (2.7)"
    Let \( S = \{ s_i : i \in I \} \) be a countable set of distinct real numbers, and let \( \{\pi_i : i \in I\} \) be a collection of real numbers satisfying

    \[
    \begin{align*}
        \pi_i &\geq 0 \quad \text{for} \quad i \in I, \\
        \sum_{i \in I} \pi_i &= 1.
    \end{align*}
    \]

    There exists a probability space \( (\Omega, \mathcal{F}, \mathbb{P}) \) and a discrete random variable \( X \) on \( (\Omega, \mathcal{F}, \mathbb{P}) \) such that the probability mass function of \( X \) is given by

    \[
    \begin{align*}
        p_X(s_i) &= \pi_i \quad \text{for} \quad i \in I, \\
        p_X(s) &= 0 \quad \text{if} \quad s \notin S.
    \end{align*}
    \]

    ??? pf "**Proof**"
        Take \( \Omega = S \), \( \mathcal{F} \) to be the set of all subsets of \( \Omega \), and

        \[
        \mathbb{P}(A) = \sum_{i: s_i \in A} \pi_i \quad \text{for} \quad A \in \mathcal{F}.
        \]

        Finally, define \( X : \Omega \to \mathbb{R} \) by \( X(\omega) = \omega \) for \( \omega \in \Omega \).
        \(\qed\)

    This theorem is very useful, since for many purposes it allows us to forget about sample spaces, event spaces, and probability measures; we need only say "let \( X \) be a random variable taking the value \( s_i \) with probability \( \pi_i \), for \( i \in I \)," and we can be sure that such a random variable exists without having to construct it explicitly.

!!! df "**Definition** (Covariance)"
    For two random variables $X, Y$, the covariance of $X, Y$, denoted $\text{Cov}(X,Y)$ is defined as
    
    $$\begin{align*}
    \text{Cov}(X, Y) = \mathbb{E}(XY) - \mathbb{E}(X)\mathbb{E}(Y)
    \end{align*}$$

!!! df "**Definition** (Uncorrelated)"
    If $X, Y$ have $\text{Cov}(X,Y) = 0$, we say that they are __*uncorrelated*__.

!!! tm "**Theorem**"
    If $X$ and $Y$ are independent then they are uncorrelated (but the converse does not hold in general).

    ??? pf "**Proof**"

!!! tm "**Theorem**"
    
    $$\begin{align*}
    \text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)
    \end{align*}$$

    $$\begin{align*}
    \text{Var}(X_1 + X_2 + \cdots + X_n) &= \sum_{i=1}^{n}\text{Var}(X_i) + 2\sum_{1\le i < j \le n}{\text{Cov}(X_i, X_j)} \\
    &= \sum_{i=1}^{n}\text{Var}(X_i) + \sum_{i \neq j}{\text{Cov}(X_i, X_j)}
    \end{align*}$$

## Random Graph

    
    

    
    