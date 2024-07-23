# Neural Network Architectures
!!! nt "**Note** (Naming Conventions)"
    Notice that when we say N-layer neural network, we do not count the input layer. Therefore, a single-layer neural network describes a network with no hidden layers (input directly mapped to output). You may also hear these networks interchangeably referred to as “Artificial Neural Networks” (ANN) or “Multi-Layer Perceptrons” (MLP).

## Representational Power
One way to look at Neural Networks with fully-connected layers is that they define a family of functions that are parameterized by the weights of the network. A natural question that arises is: Are there functions that cannot be modeled with a Neural Network?

!!! tm "**Lemma** (Representational Power of Neuron Network)"
    Given any continuous function $f(x)$ and some $\epsilon>0$, there exists a Neural Network $g(x)$ with one hidden layer (with a reasonable choice of non-linearity, e.g. sigmoid) such that $\forall x,\left| f(x)−g(x) \right| < \epsilon$. In other words, the neural network can approximate any continuous function.

!!! nt "**Note** (Why Not One Hidden Layer?)"
    If one hidden layer suffices to approximate any function, why use more layers and go deeper? The answer is that the fact that a two-layer Neural Network is a universal approximator is, while mathematically cute, a relatively weak and useless statement in practice. The fact that deeper networks (with multiple hidden layers) can work better than a single-hidden-layer networks is an empirical observation, despite the fact that their representational power is equal.

!!! nt "**Note** (More Layers or Not?)"
    In practice it is often the case that 3-layer neural networks will outperform 2-layer nets, but going even deeper (4,5,6-layer) rarely helps much more. This is in stark contrast to Convolutional Networks, where depth has been found to be an extremely important component for a good recognition system (e.g. on order of 10 learnable layers). One argument for this observation is that images contain hierarchical structure (e.g. faces are made up of eyes, which are made up of edges, etc.), so several layers of processing make intuitive sense for this data domain.

!!! im "**Important Note** (Setting Number of Layers and Their Sizes)"
    How do we decide on what architecture to use when faced with a practical problem? First, note that as we increase the size and number of layers in a Neural Network, neurons can collaborate to express many more different functions. 

    <figure markdown="span">
    ![Image title](https://cs231n.github.io/assets/nn1/layer_sizes.jpeg){ width="600" }
    <figcaption>A Binary Classification Problem</figcaption>
    </figure>

    However, **overfitting** occurs if the NN has too much capability. But this does not mean we should choose a smaller model to avoid overfitting - there are many other preferred ways to prevent overfitting in Neural Networks such as L2 regularization, dropout, input noise. In practice, it is always better to use these methods to control overfitting instead of the number of neurons. The regularization strength is the preferred way to control the overfitting of a neural network. We can look at the results achieved by three different settings:

    <figure markdown="span">
    ![Image title](https://cs231n.github.io/assets/nn1/reg_strengths.jpeg){ width="600" }
    <figcaption></figcaption>
    </figure>

    Changing the regularization strength makes its final decision regions smoother with a higher regularization. You can play this [<u>here</u>](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html).

## Data Preprocessing
There are three common forms of data preprocessing a data matrix `X`, where we will assume that `X` is of size `[N x D]` (`N` is the number of data, `D` is their dimensionality).
!!! st "**Strategy** (Mean Subtraction)"
    **Mean subtraction** is the most common form of preprocessing. It involves subtracting the mean across every individual *feature* in the data, and has the geometric interpretation of centering the cloud of data around the origin along every dimension. In numpy, this operation would be implemented as: `X -= np.mean(X, axis=0)`. With images specifically, for convenience it can be common to subtract a single value from all pixels (e.g. `X -= np.mean(X)`), or to do so separately across the three color channels.

!!! st "**Strategy** (Normalization)"
    **Normalization** refers to normalizing the data dimensions so that they are of approximately the same scale. There are two common ways of achieving this normalization. One is to divide each dimension by its standard deviation, once it has been zero-centered: `X /= np.std(X, axis=0)`. Another form of this preprocessing normalizes each dimension so that the min and max along the dimension is -1 and 1 respectively. It only makes sense to apply this preprocessing if you have a reason to believe that different input features have different scales (or units), but they should be of approximately equal importance to the learning algorithm. In case of images, the relative scales of pixels are already approximately equal (and in range from 0 to 255), so it is not strictly necessary to perform this additional preprocessing step.

<figure markdown="span">
![Image title](https://cs231n.github.io/assets/nn2/prepro1.jpeg){ width="600" }
<figcaption>Common Data Preprocessing Pipeline</figcaption>
</figure>

!!! st "**Strategy** (PCA and Whitening)"
    PCA and Whitening is another form of preprocessing. In this process, the data is first centered as described above. Then, we can compute the covariance matrix that tells us about the correlation structure in the data: <br>
    
    - [ ] PCA and Whitening

    <figure markdown="span">
    ![Image title](https://cs231n.github.io/assets/nn2/prepro2.jpeg){ width="600" }
    <figcaption></figcaption>
    </figure>

??? eg "**Example** (Visualizing Whitened Images)"
    The training set of CIFAR-10 is of size 50,000 x 3072, where every image is stretched out into a 3072-dimensional row vector. We can then compute the [3072 x 3072] covariance matrix and compute its SVD decomposition (which can be relatively expensive). What do the computed eigenvectors look like visually? An image might help:

    <figure markdown="span">
    ![Image title](https://cs231n.github.io/assets/nn2/cifar10pca.jpeg){ width="800" }
    <figcaption>PCA / Whitening</figcaption>
    </figure>

!!! im "**Important Note** (In Practice)"
    We mention PCA/Whitening in these notes for completeness, but these transformations are not used with Convolutional Networks. However, it is very important to zero-center the data, and it is common to see normalization of every pixel as well.

!!! wr "**Warning** (Only Preprocess Training Data)"
    Any preprocessing statistics (e.g. the data mean) must only be computed on the training data, and then applied to the validation / test data.

## Weight Initialization
We have seen how to construct a Neural Network architecture, and how to preprocess the data. Before we can begin to train the network we have to initialize its parameters.

!!! wr "**Warning** (Pitfall: All Zero Initialization)"
    At the end of training, with a proper data normalization it is reasonable to assume that approximately half of the weights will be positive and half of them will be negative. We should not set all parameters to be 0 because if every neuron in the network computes the same output, then they will also all compute the same gradients during backpropagation and undergo the exact same parameter updates.

!!! st "**Strategy** (Small Random Numbers Initialization)"
    It is common to initialize the weights of the neurons to small numbers and refer to doing so as symmetry breaking. The implementation for one weight matrix might look like `W = 0.01* np.random.randn(D,H)`, where `randn` samples from a zero mean, unit standard deviation Gaussian. 

!!! wr "**Warning** (Limitation of Small Random Numbers Initialization)"
    It’s not necessarily the case that smaller numbers will work strictly better. For example, a Neural Network layer that has very small weights will during backpropagation compute very small gradients on its data (since this gradient is proportional to the value of the weights). This could greatly diminish the “gradient signal” flowing backward through a network, and could become a concern for deep networks.

!!! st "**Strategy** (Calibrating The Variances with $\frac{1}{\sqrt{n}}$)"
    One problem with the above suggestion is that the distribution of the outputs from a randomly initialized neuron has a variance that grows with the number of inputs. It turns out that we can normalize the variance of each neuron’s output to 1 by scaling its weight vector by the square root of its fan-in (i.e. its number of inputs). That is, the recommended heuristic is to initialize each neuron’s weight vector as: `w = np.random.randn(n) / sqrt(n)`, where `n` is the number of its inputs. This ensures that all neurons in the network initially have approximately the same output distribution and empirically improves the rate of convergence.<br>

    - [ ] Proof of this Strategy

!!! st "**Strategy** (Sparse Initialization)"
    Another way to address the uncalibrated variances problem is to set all weight matrices to zero, but to break symmetry every neuron is randomly connected (with weights sampled from a small gaussian as above) to a fixed number of neurons below it. A typical number of neurons to connect to may be as small as 10.

!!! im "**Important Note** (Initializing The Biases)"
    It is common to initialize the biases to be zero. For ReLU non-linearities, some people like to use small constant value such as 0.01 because this ensures that all ReLU units fire in the beginning and therefore obtain and propagate some gradient. However, it is not clear if this is useful and it is more common to simply use 0 bias initialization.

!!! im "**Important Note** (In Practice)"
    The current recommendation is to use ReLU units and use the `w = np.random.randn(n) * sqrt(2.0/n)`, as discussed in [<u>He et al.</u>](http://arxiv-web3.library.cornell.edu/abs/1502.01852).

!!! st "**Strategy** (Batch Normalization)"
    A recently developed technique by Ioffe and Szegedy called [<u>Batch Normalization</u>](https://arxiv.org/abs/1502.03167) alleviates a lot of headaches with properly initializing neural networks by explicitly forcing the activations throughout a network to take on a unit gaussian distribution at the beginning of the training.

## Regularization
There are several ways of controlling the capacity of Neural Networks to prevent overfitting:
!!! st "**Strategy** (L2 Regularization)"
    For every weight $w$ in the network, we add the term $\frac{1}{2}\lambda w^2$ to the objective, where $\lambda$ is the regularization strength.

!!! st "**Strategy** (L1 Regularization)"
    For each weight $w$ we add the term $\lambda \left| w \right|$ to the objective.

!!! nt "**Note** (L2 v.s. L1)"
    In practice, if you are not concerned with explicit feature selection, L2 regularization can be expected to give superior performance over L1.

!!! st "**Strategy** (Elastic Net Regularization)"
    Elastic net regularization is a combination of L1 & L2 regularization: $\lambda_1 \left| w \right| + \lambda_2 w^2$.

!!! st "**Strategy** (Max Norm Constraints)"
    Max norm put constraints to enforce an absolute upper bound on the magnitude of the weight vector for every neuron and use projected gradient descent to enforce the constraint. In practice, this corresponds to performing the parameter update as normal, and then enforcing the constraint by clamping the weight vector `\(\vec{w}\)` of every neuron to satisfy `\(\|\vec{w}\|_2 < c\)`. Typical values of `c` are on orders of 3 or 4. Some people report improvements when using this form of regularization. One of its appealing properties is that network cannot "explode" even when the learning rates are set too high because the updates are always bounded.

!!! st "**Strategy** (Dropout)"
    Dropout is an extremely effective, simple and recently introduced regularization technique by Srivastava et al. in [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/v15/srivastava14a.html) that complements the other methods (L1, L2, maxnorm). While training, dropout is implemented by only keeping a neuron active with some probability \(p\) (a hyperparameter), or setting it to zero otherwise.

    <figure markdown="span">
    ![Image title](https://cs231n.github.io/assets/nn2/dropout.jpeg){ width="400" }
    <figcaption></figcaption>
    </figure>

!!! im "**Important Note** (In Practice)"
    It is most common to use a single, global L2 regularization strength that is cross-validated. It is also common to combine this with dropout applied after all layers. The value of $p=0.5$ is a reasonable default, but this can be tuned on validation data.
    
    
    
    
    
    
    

    
    
    
    
    
    