# Generative Learning Algorithms

## Discriminative v.s. Generative
!!! df "**Definition** (Discriminative Learning Algorithms)"
    Algorithms that try to learn $p(y\mid x)$ directly (such as logistic regression),
    or algorithms that try to learn mappings directly from the space of inputs $X$
    to the labels $\{0, 1\}$, (such as the perceptron algorithm) are called _Discriminative Learning Algorithms_.

!!! nt "**Note** (Intuition of Discriminative Learning Algorithms)"
    Consider a classification problem in which we want to learn to distinguish
    between elephants ($y = 1$) and dogs ($y = 0$), based on some features of
    an animal. Given a training set, an algorithm like logistic regression or
    the perceptron algorithm (basically) tries to find a straight line—that is, a
    decision boundary—that separates the elephants and dogs. Then, to classify
    a new animal as either an elephant or a dog, it checks on which side of the
    decision boundary it falls, and makes its prediction accordingly.

!!! df "**Definition** (Generative Learning Algorithms)"
    Instead of learning $p(y\mid x)$, _Generative Learning Algorithms_ tries to model $p(x\mid y)$ (and $p(y)$). For instance, if $y$ indicates whether an example is a dog ($0$) or an elephant ($1$), then $p(x\mid y = 0)$ models the distribution of dogs’
    features, and $p(x\mid y = 1)$ models the distribution of elephants’ features.

!!! nt "**Note** (Intuition of Generative Learning Algorithms)"
    First, looking at elephants, we can build a
    model of what elephants look like. Then, looking at dogs, we can build a
    separate model of what dogs look like. Finally, to classify a new animal, we
    can match the new animal against the elephant model, and match it against
    the dog model, to see whether the new animal looks more like the elephants
    or more like the dogs we had seen in the training set. 

!!! df "**Definition** (Posterior Probability[^1])"
    In [variational Bayesian methods](https://en.wikipedia.org/wiki/Variational_Bayesian_methods), the posterior probability is the probability of the parameters \( \theta \) given the evidence \( X \), and is denoted \( p(\theta \mid X) \).
    It contrasts with the likelihood function, which is the probability of the evidence given the parameters: \( p(X \mid \theta) \).

!!! nt "**Note** (Posterior Probability v.s. Likelihood Function[^1])"
    Given a prior belief that a probability distribution function is \( p(\theta) \) and that the observations \( x \) have a likelihood \( p(x \mid \theta) \), then the posterior probability is defined as

    \[ p(\theta \mid x) = \frac{p(x \mid \theta)}{p(x)} p(\theta), \]

    where $p(\theta)$ is called _Prior Probability_, and $\( p(x) \) is the normalizing constant and is calculated as

    \[ p(x) = \int p(x \mid \theta) p(\theta) d\theta \]

    for continuous \( \theta \), or by summing \( p(x \mid \theta) p(\theta) \) over all possible values of \( \theta \) for discrete \( \theta \).

    The posterior probability is therefore proportional to the product Likelihood \(\cdot\) Prior probability.


??? eg "**Example** (Example of Posterior Probability[^1])"
    Suppose there is a school with 60% boys and 40% girls as students. The girls wear trousers or skirts in equal numbers; all boys wear trousers. An observer sees a (random) student from a distance; all the observer can see is that this student is wearing trousers. What is the probability this student is a girl? The correct answer can be computed using Bayes' theorem.

    The event \( G \) is that the student observed is a girl, and the event \( T \) is that the student observed is wearing trousers. To compute the posterior probability \( P(G \mid T) \), we first need to know:

    - \( P(G) \), or the probability that the student is a girl regardless of any other information. Since the observer sees a random student, meaning that all students have the same probability of being observed, and the percentage of girls among the students is 40%, this probability equals 0.4.
    - \( P(B) \), or the probability that the student is not a girl (i.e. a boy) regardless of any other information (\( B \) is the complementary event to \( G \)). This is 60%, or 0.6.
    - \( P(T \mid G) \), or the probability of the student wearing trousers given that the student is a girl. As they are as likely to wear skirts as trousers, this is 0.5.
    - \( P(T \mid B) \), or the probability of the student wearing trousers given that the student is a boy. This is given as 1.
    - \( P(T) \), or the probability of a (randomly selected) student wearing trousers regardless of any other information. Since \( P(T) = P(T \mid G)P(G) + P(T \mid B)P(B) \) (via the law of total probability), this is \( P(T) = 0.5 \times 0.4 + 1 \times 0.6 = 0.8 \).

    Given all this information, the **posterior probability** of the observer having spotted a girl given that the observed student is wearing trousers can be computed by substituting these values in the formula:

    \[ P(G \mid T) = \frac{P(T \mid G)P(G)}{P(T)} = \frac{0.5 \times 0.4}{0.8} = 0.25 \]

    [^1]: [_Wikipedia: Posterior Probability_](https://en.wikipedia.org/wiki/Posterior_probability)    

!!! im "**Important Note** (GLA Logistic)"
    When using GLAs to do a classification, we first use MLE to learn the joint probability $p(x,y)$, which is to learn $p(y=0)$ and $p(y=1)$, the **Class Priors**, and $p(x\mid y=0)$ and $p(x\mid y=1)$. Then, we can use Bayes rule to derive the posterior distribution:
    
    $$\begin{align*}
    p(y\mid x) &= \frac{p(x\mid y)p(y)}{p(x)} \\
               &= \frac{p(x\mid y)p(y)}{p(x\mid y=1)p(y=1) + p(x\mid y=0)p(y=0)}
    \end{align*}$$

    We can calculate both situation where $y=1$ or $y=0$ and compare which has a larger possibility. This is how we "compare" which models best fit $x$.

    Note that we don't actually need to calculate $p(x)$, as

    \[
    \arg \max_{y} p(y \mid x) = \arg \max_{y} \frac{p(x \mid y) p(y)}{p(x)} = \arg \max_{y} p(x \mid y) p(y).
    \]

There are two different types of generative learning models: _Gaussian Discriminant Analysis Model_ and _Naive Bayes_. We’ll introduce them in the following.

## Gaussian Discriminant Analysis (GDA)

!!! df "**Definition** (Multivariate Normal Distribution)"
    The multivariate normal distribution in \( d \)-dimensions, also called the multivariate Gaussian distribution, is parameterized by a **mean vector** \( \mu \in \mathbb{R}^d \) and a **covariance matrix** \( \Sigma \in \mathbb{R}^{d \times d} \), where \( \Sigma \geq 0 \) is symmetric and positive semi-definite. Also written \( \mathcal{N}(\mu, \Sigma) \), its density is given by:

    \[
    p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left( -\frac{1}{2}(x - \mu)^T \Sigma^{-1}(x - \mu) \right).
    \]

    In the equation above, \( |\Sigma| \) denotes the determinant of the matrix \( \Sigma \).

    For a random variable \( X \) distributed \( \mathcal{N}(\mu, \Sigma) \), the mean is (unsurprisingly) given by \( \mu \):

    \[
    \mathbb{E}[X] = \int x \, p(x; \mu, \Sigma) \, dx = \mu.
    \]

    The **covariance** of a vector-valued random variable \( Z \) is defined as \( \mathrm{Cov}(Z) = \mathbb{E}[(Z - \mathbb{E}[Z])(Z - \mathbb{E}[Z])^T] \). This generalizes the notion of the variance of a scalar.

    The covariance can also be defined as \( \mathrm{Cov}(Z) = \mathbb{E}[ZZ^T] - (\mathbb{E}[Z])(\mathbb{E}[Z])^T \). (You should be able to prove to yourself that these two definitions are equivalent.) If \( X \sim \mathcal{N}(\mu, \Sigma) \), then

    \[
    \mathrm{Cov}(X) = \Sigma.
    \]

!!! df "**Definition** (Gaussian Discriminant Analysis Model)"
    When we have a classification problem in which the input features \( x \) are continuous-valued random variables, we can then use the Gaussian Discriminant Analysis (GDA) model, which models \( p(x \mid y) \) using a multivariate normal distribution. The model is:

    \[
    \begin{align*}
    y &\sim \text{Bernoulli}(\phi) \\
    x \mid y = 0 &\sim \mathcal{N}(\mu_0, \Sigma) \\
    x \mid y = 1 &\sim \mathcal{N}(\mu_1, \Sigma)
    \end{align*}
    \]

    Writing out the distributions, this is:

    \[
    \begin{align*}
    p(y) &= \phi^y (1 - \phi)^{1 - y} \\
    p(x \mid y = 0) &= \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu_0)^T \Sigma^{-1} (x - \mu_0) \right) \\
    p(x \mid y = 1) &= \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu_1)^T \Sigma^{-1} (x - \mu_1) \right)
    \end{align*}
    \]

    Here, the parameters of our model are \( \phi \), \( \Sigma \), \( \mu_0 \) and \( \mu_1 \). (Note that while there are two different mean vectors \( \mu_0 \) and \( \mu_1 \), this model is usually applied using only one covariance matrix \( \Sigma \).) The log-likelihood of the data is given by

    \[
    \begin{align*}
    \ell (\phi, \mu_0, \mu_1, \Sigma) &= \log \prod_{i=1}^n p(x^{(i)}, y^{(i)} ; \phi, \mu_0, \mu_1, \Sigma) \\
    &= \log \prod_{i=1}^n p(x^{(i)} \mid y^{(i)} ; \mu_0, \mu_1, \Sigma) p(y^{(i)} ; \phi).
    \end{align*}
    \]

    <figure markdown="span">
    ![Image title](https://stanford.edu/~shervine/teaching/cs-229/illustrations/generative-model.png?df0642cec6e99ac162cd4848d26f41c3){ width="400" }
    <figcaption>Probability Distributions of The Data</figcaption>
    </figure>

## Naive Bayes (NB)

For our motivating example, consider building an email spam filter. Here, we wish to classify messages according to whether
they are unsolicited commercial (spam) email, or non-spam email. Classifying emails is one example of a broader set of problems called text
classification.

Let's say we have a training set (a set of emails labeled as spam or non-spam). We'll begin our construction of our spam filter by specifying the features \( x_j \) used to represent an email.

We will represent an email via a feature vector whose length is equal to the number of distinct words in the training set. Specifically, if an email contains the \( j \)-th word of the dictionary, then we will set \( x_j = 1 \); otherwise, we let \( x_j = 0 \). For instance, the vector

\[
x = \begin{bmatrix}
1 \\
0 \\
0 \\
\vdots \\
1 \\
\vdots \\
0 \\
\end{bmatrix}
\]

is used to represent an email that contains the words "a" and "buy." The set of words encoded into the feature vector is called the **vocabulary**, so the dimension of \( x \) is equal to the size of the vocabulary.

Having chosen our feature vector, we now want to build a generative model. So, we have to model \( p(x \mid y) \). But if we have, say, a vocabulary of 50000 words, then \( x \in \{0, 1\}^{50000} \) (i.e. a 50000-dimensional vector of 0's and 1's), and if we were to model \( x \) explicitly with a multinomial distribution over the \( 2^{50000} \) possible outcomes, then we'd end up with a \( (2^{50000} - 1) \)-dimensional parameter vector. This is clearly too many parameters.

!!! df "**Definition** (Naive Bayes Assumption)"
    To model \( p(x \mid y) \), we make a very strong assumption: \( x_i \)'s are conditionally independent given \( y \). This assumption is called the _Naive Bayes Assumption_, and the resulting algorithm is called the _Naive Bayes classifier_. Generally. 
    
    \[
    P[X^{(1)} \ldots X^{(n)} \mid Y] = \prod_{i=1}^{n} P[X^{(i)} \mid Y]
    \]

??? eg "**Example** (Example of NB Assumption)"
    If \( y = 1 \) means spam email; "buy" is word 2087 and "price" is word 39831; then we are assuming that if I tell you \( y = 1 \) (that a particular piece of email is spam), then knowledge of \( x_{2087} \) (knowledge of whether "buy" appears in the message) will have no effect on your beliefs about the value of \( x_{39831} \) (whether "price" appears). More formally, this can be written \( p(x_{2087} \mid y) = p(x_{2087} \mid y, x_{39831}) \). (Note that this is not the same as saying that \( x_{2087} \) and \( x_{39831} \) are independent, which would have been written \( p(x_{2087}) = p(x_{2087} \mid x_{39831}) \); rather, we are only assuming that \( x_{2087} \) and \( x_{39831} \) are conditionally independent given \( y \).)

    We now have:

    \[
    \begin{align*}
    p(x_1, \ldots, x_{50000} \mid y) &= p(x_1 \mid y) p(x_2 \mid y, x_1) p(x_3 \mid y, x_1, x_2) \cdots p(x_{50000} \mid y, x_1, \ldots, x_{49999}) \\
    &= p(x_1 \mid y) p(x_2 \mid y) p(x_3 \mid y) \cdots p(x_{50000} \mid y) \\
    &= \prod_{j=1}^{50000} p(x_j \mid y)
    \end{align*}
    \]

    The first equality simply follows from the usual properties of probabilities, and the second equality used the NB assumption. We note that even though the Naive Bayes assumption is an extremely strong assumption, the resulting algorithm works well on many problems.

!!! df "**Model** (Naive Bayes Classifier)"
    Our model is parameterized by \( \phi_{j \mid y=1} = p(x_j = 1 \mid y = 1) \), \( \phi_{j \mid y=0} = p(x_j = 1 \mid y = 0) \), and \( \phi_y = p(y = 1) \). As usual, given a training set \( \{ (x^{(i)}, y^{(i)}) ; i = 1, \ldots, n \} \), we can write down the joint likelihood of the data:

    \[
    \mathcal{L}(\phi_y, \phi_{j \mid y=0}, \phi_{j \mid y=1}) = \prod_{i=1}^n p(x^{(i)}, y^{(i)}).
    \]

    Maximizing this with respect to \( \phi_y \), \( \phi_{j \mid y=0} \), and \( \phi_{j \mid y=1} \) gives the maximum likelihood estimates:
    
    \[
    \begin{align*}
    \phi_{j \mid y=1} &= \frac{\sum_{i=1}^n 1 \{ x_j^{(i)} = 1 \land y^{(i)} = 1 \}}{\sum_{i=1}^n 1 \{ y^{(i)} = 1 \}} \\
    \phi_{j \mid y=0} &= \frac{\sum_{i=1}^n 1 \{ x_j^{(i)} = 1 \land y^{(i)} = 0 \}}{\sum_{i=1}^n 1 \{ y^{(i)} = 0 \}} \\
    \phi_y &= \frac{\sum_{i=1}^n 1 \{ y^{(i)} = 1 \}}{n}
    \end{align*}
    \]

    The parameters have a very natural interpretation. For instance, \( \phi_{j \mid y=1} \) is just the fraction of the spam ( \( y = 1 \) ) emails in which word \( j \) does appear.

    Having fit all these parameters, to make a prediction on a new example with features \( x \), we then simply calculate

    \[
    \begin{align*}
    p(y = 1 \mid x) &= \frac{p(x \mid y = 1) p(y = 1)}{p(x)} \\
    &= \frac{\left( \prod_{j=1}^d p(x_j \mid y = 1) \right) p(y = 1)}{\left( \prod_{j=1}^d p(x_j \mid y = 1) \right) p(y = 1) + \left( \prod_{j=1}^d p(x_j \mid y = 0) \right) p(y = 0)},
    \end{align*}
    \]

    and pick whichever class has the higher posterior probability.

!!! rm "**Remark** (NB for Multiple Classes)"
    We note that while we have developed the Naive Bayes algorithm mainly for the case of problems where the features \( x_j \) are binary-valued, the generalization to where \( x_j \) can take values in \( \{1, 2, \ldots, k_j \} \) is straightforward. Here, we would simply model \( p(x_j \mid y) \) as multinomial rather than as Bernoulli. Indeed, even if some original input attribute (say, the living area of a house, as in our earlier example) were continuous-valued, it is quite common to **discretize** it—that is, turn it into a small set of discrete values—and apply Naive Bayes. For instance, if we use some feature \( x_j \) to represent living area, we might discretize the continuous values as follows:

    \[
    \begin{array}{c|c|c|c|c|c}
    \text{Living area (sq. feet)} & < 400 & 400-800 & 800-1200 & 1200-1600 & > 1600 \\
    x_i & 1 & 2 & 3 & 4 & 5 \\
    \end{array}
    \]

    Thus, for a house with living area 890 square feet, we would set the value of the corresponding feature \( x_j \) to 3. We can then apply the Naive Bayes algorithm, and model \( p(x_j \mid y) \) with a multinomial distribution, as described previously. When the original, continuous-valued attributes are not well-modeled by a multivariate normal distribution, discretizing the features and using Naive Bayes (instead of GDA) will often result in a better classifier.

!!! im "**Important Note** (Comparing GDA and NB)"
    GDA: <br>

    - continuous $x \in \mathbb{R}^d$ <br>
    - $p(x\mid y) \sim \mathcal{N}(\mu_y, \Sigma)$ <br>
    - $p(y\mid x) = \frac{1}{1+\text{exp}(-\theta^Tx)}$ <br>
    
    NB: <br>
    
    - discrete $x$ <br>
    - conditional Independence Assumption: $p(x_j \mid y, x_k) = p(x_j\mid y)$ <br>
    - Bernoulli Event Model: $p(x_j \mid y) \sim \text{Bernoulli}[x_j \text{is jth word in vocabulary}]$ <br>
    - multinomial Event Model: $p(x_j \mid y) \sim \text{Multinomial}[x_j \text{is jth word in message}]$

## Laplace Smoothing

The Naive Bayes algorithm as we have described it will work fairly well
for many problems, but there is a simple change that makes it work much
better, especially for text classification.

- [ ] Laplace Smoothing 

    
    








    
    
    
    





