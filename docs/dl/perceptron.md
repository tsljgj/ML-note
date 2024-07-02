# Multiplayer Perceptrons

Linearity is not always the case for models in reality. We sometimes want to find a much more complicated relations between features and targets. 

## Multilayer Perceptrons
!!! df "**Model** (Multilayer Perceptron)"
    We can overcome the limitations of linear models by incorporating one or more hidden layers. This architecture is commonly called a _multilayer perceptron_ (MLP). 
    
    <figure markdown="span">
    ![Image title](https://d2l.ai/_images/mlp.svg){ width="400" }
    <figcaption>Multilayer Perceptron</figcaption>
    </figure>

    As before, we denote by the matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ a minibatch of $n$ examples where each example has $d$ inputs (features). For a one-hidden-layer MLP whose hidden layer has $h$ hidden units, we denote by $\mathbf{H} \in \mathbb{R}^{n \times h}$ the outputs of the hidden layer, which are **hidden representations**. Since the hidden and output layers are both fully connected, we have hidden-layer weights $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ and biases $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$ and output-layer weights $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ and biases $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$. This allows us to calculate the outputs $\mathbf{O} \in \mathbb{R}^{n \times q}$ of the one-hidden-layer MLP as follows:

    $$\begin{align*}
    \mathbf{H} &= \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)} \\
    \mathbf{O} &= \mathbf{H} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}
    \end{align*}$$

    However, this is not enough as using two layers which are computed linearly will still result in a linear model.

    To see this formally we can just collapse out the hidden layer in the above definition, yielding an equivalent single-layer model with parameters $\mathbf{W} = \mathbf{W}^{(1)} \mathbf{W}^{(2)}$ and $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$:

    $$
    \mathbf{O} = \left(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}\right) \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}
    $$

    In order to realize the potential of multilayer architectures, we need one more key ingredient: a nonlinear **activation function** $\sigma$ to be applied to each hidden unit following the affine transformation. The outputs of activation functions $\sigma(\cdot)$ are called **activations**. In general, with activation functions in place, it is no longer possible to collapse our MLP into a linear model:

    $$\begin{align*}
    \mathbf{H} &= \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}) \\
    \mathbf{O} &= \mathbf{H} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}
    \end{align*}$$

## Activation Functions
!!! df "**Definition** (ReLU Function)"
    _Rectified Linear Unit_ (ReLU) provides a very simple nonlinear transformation. Given an element $x$, the function is defined as the maximum of that element and $0$:

    $$
    \text{ReLU}(x) = \max(x, 0)
    $$

    <figure markdown="span">
    ![Image title](https://d2l.ai/_images/output_mlp_76f463_18_0.svg){ width="400" }
    <figcaption>ReLU Function</figcaption>
    </figure>

    When the input is negative, the derivative of the ReLU function is 0, and when the input is positive, the derivative of the ReLU function is 1. Note that the ReLU function is not differentiable when the input takes value precisely equal to 0. In these cases, we default to the left-hand-side derivative and say that the derivative is 0 when the input is 0. We can get away with this because the input may never actually be zero.

    <figure markdown="span">
    ![Image title](https://d2l.ai/_images/output_mlp_76f463_33_0.svg){ width="400" }
    <figcaption>Derivative of ReLU Function</figcaption>
    </figure>

!!! df "**Definition** (pReLU)"
    The parametrized ReLU (pReLU) adds a linear term to ReLU, so some information still gets through, even when the argument is negative:
    
    $$\begin{align*}
    \text{pReLU}(x) = \max(0,x) + \alpha\min(0,x)
    \end{align*}$$

!!! df "**Definition** (Sigmoid)"
    The sigmoid function transforms those inputs whose values lie in the domain $\mathbb{R}$, to outputs that lie on the interval $(0, 1)$. For that reason, the sigmoid is often called a squashing function: it squashes any input in the range $(-\infty, \infty)$ to some value in the range $(0, 1)$:

    $$
    \text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.
    $$

    Below, we plot the sigmoid function. Note that when the input is close to 0, the sigmoid function approaches a linear transformation.
    
    <figure markdown="span">
    ![Image title](https://d2l.ai/_images/output_mlp_76f463_48_0.svg){ width="400" }
    <figcaption>Sigmoid Function</figcaption>
    </figure>

    The derivative of the sigmoid function is given by the following equation:

    $$
    \frac{d}{dx} \text{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \text{sigmoid}(x) \left(1 - \text{sigmoid}(x)\right).
    $$

    The derivative of the sigmoid function is plotted below. Note that when the input is 0, the derivative of the sigmoid function reaches a maximum of 0.25. As the input diverges from 0 in either direction, the derivative approaches 0.

    <figure markdown="span">
    ![Image title](https://d2l.ai/_images/output_mlp_76f463_63_0.svg){ width="400" }
    <figcaption>Derivative of Sigmoid Function</figcaption>
    </figure>

!!! df "**Definition** (Tanh)"  
    Like the sigmoid function, the tanh (hyperbolic tangent) function also squashes its inputs, transforming them into elements on the interval between $(-1,1)$:

    $$
    \tanh(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.
    $$

    We plot the tanh function below. Note that as input nears 0, the tanh function approaches a linear transformation. Although the shape of the function is similar to that of the sigmoid function, the tanh function exhibits point symmetry about the origin of the coordinate system.

    <figure markdown="span">
    ![Image title](https://d2l.ai/_images/output_mlp_76f463_78_0.svg){ width="400" }
    <figcaption>Tanh Function</figcaption>
    </figure>

    The derivative of the tanh function is:

    $$
    \frac{d}{dx} \tanh(x) = 1 - \tanh^2(x).
    $$

    It is plotted below. As the input nears 0, the derivative of the tanh function approaches a maximum of 1. And as we saw with the sigmoid function, as input moves away from 0 in either direction, the derivative of the tanh function approaches 0.

    <figure markdown="span">
    ![Image title](https://d2l.ai/_images/output_mlp_76f463_93_0.svg){ width="400" }
    <figcaption>Derivative of Tanh Function</figcaption>
    </figure>