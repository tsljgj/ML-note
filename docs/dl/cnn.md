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
    
## Architecture Overview
!!! im "**Important Note** (Layers in ConvNets)"
    We use three main types of layers to build ConvNet architectures: Convolutional Layer, Pooling Layer, and Fully-Connected Layer. We will stack these layers to form a full ConvNet architecture.

??? eg "**Example** (ConvNet Architecture for CIFAR-10 Classification)"
    - **INPUT Layer** [32x32x3] will hold the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B.
    - **CONV Layer** will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. This may result in volume such as [32x32x12] if we decided to use 12 filters.
    - **RELU Layer** will apply an elementwise activation function, such as the $\max(0,x)$ thresholding at zero. This leaves the size of the volume unchanged ([32x32x12]).
    - **POOL Layer** will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x12].
    - **FC Layer** (i.e. fully-connected) will compute the class scores, resulting in volume of size [1x1x10], where each of the 10 numbers correspond to a class score, such as among the 10 categories of CIFAR-10.

    <figure markdown="span">
    ![Image title](https://cs231n.github.io/assets/cnn/convnet.jpeg){ width="600" }
    <figcaption>Activations of an ConvNet Architecture</figcaption>
    </figure>

## Convolutional Layer
!!! df "**Definition** (Filter - Receptive Fields)"
    The _Filter_, or the Receptive Field, in the context of CNN, is a $F \times F \times 3$ square with which we use to multiply local regions in the image. 

!!! im "**Important Note** (Intuition about Filters)"
    Each filter is looking for a specific feature in the picture.

!!! df "**Definition** (Stride)"
    _Stride_ is the number of the pixel jumped when the filters slide. When the stride is 1 then we move the filters one pixel at a time. When the stride is 2 (or uncommonly 3 or more, though this is rare in practice) then the filters jump 2 pixels at a time as we slide them around. This will produce smaller output volumes spatially.

!!! im "**Important Note** (Why Stride 1)"
    Why use stride of 1 in CONV? Smaller strides work better in practice. Additionally, as already mentioned stride 1 allows us to leave all spatial down-sampling to the POOL layers, with the CONV layers only transforming the input volume depth-wise.

!!! df "**Definition** (Zero-Padding)"
    The _zero-padding_ is a boarder around the input volume that only has element 0. Sometimes it will be convenient to pad the input volume with zeros around the border. The size of this zero-padding is a hyperparameter. The nice feature of zero padding is that it will allow us to control the spatial size of the output volumes

!!! im "**Important Note** (Why Padding?)"
    Why use padding? In addition to keeping the spatial sizes constant after CONV, doing this actually improves performance. If the CONV layers were to not zero-pad the inputs and only perform valid convolutions, then the size of the volumes would reduce by a small amount after each CONV, and the information at the borders would be “washed away” too quickly.

!!! df "**Definition** (Computing Output volume)"
    The Conv Layer:

    - Accepts a volume of size \( W_1 \times H_1 \times D_1 \)
    - Requires four hyperparameters:
    - Number of filters \( K \),
    - their spatial extent \( F \),
    - the stride \( S \),
    - the amount of zero padding \( P \).
    - Produces a volume of size \( W_2 \times H_2 \times D_2 \) where:
    - \( W_2 = \left(\frac{W_1 - F + 2P}{S}\right) + 1 \)
    - \( H_2 = \left(\frac{H_1 - F + 2P}{S}\right) + 1 \) (i.e. width and height are computed equally by symmetry)
    - \( D_2 = K \)
    - With parameter sharing, it introduces \( F \cdot F \cdot D_1 \) weights per filter, for a total of \( (F \cdot F \cdot D_1) \cdot K \) weights and \( K \) biases.
    - In the output volume, the \( d \)-th depth slice (of size \( W_2 \times H_2 \)) is the result of performing a valid convolution of the \( d \)-th filter over the input volume with a stride of \( S \), and then offset by \( d \)-th bias.

    A common setting of the hyperparameters is \( F = 3 \), \( S = 1 \), \( P = 1 \).

!!! im "**Important Note** (Convolution Demo)"
    Below is a running demo of a CONV layer. The input volume is of size \( W_1 = 5 \), \( H_1 = 5 \), \( D_1 = 3 \), and the CONV layer parameters are \( K = 2 \), \( F = 3 \), \( S = 2 \), \( P = 1 \). Therefore, the output volume size has spatial size \( (5 - 3 + 2)/2 + 1 = 3 \). The visualization below iterates over the output activations (green), and shows that each element is computed by **elementwise multiplying the highlighted input (blue) with the filter (red), summing it up, and then offsetting the result by the bias**.<br><br>
    
    <div class="fig figcenter fighighlight">
    <iframe src="https://cs231n.github.io/assets/conv-demo/index.html" width="100%" height="700px;" style="border:none;"></iframe>
    <div class="figcaption"></div>
    </div>

!!! im "**Important Note** (Implementation as Matrix Multiplication)"
    A common implementation pattern of the CONV layer is to formulate the forward pass of a convolutional layer as one big matrix multiply as follows:<br><br>
    
    The local regions (blocks that have the same shape as the filter) in the input image are stretched out into columns in an operation commonly called **im2col**. For example, if the input is [227x227x3] and it is to be convolved with 11x11x3 filters at stride 4, then we would take blocks of shape [11x11x3] in the input and stretch each block into a column vector of size 11*11*3 = 363. Iterating this process in the input at stride of 4 gives $((227-11)/4+1)^2$ = 3025 blocks, leading to an output matrix $X_{col}$ of _im2col_ of size [363 x 3025].<br>
    <br>
    Remember that we are to multiply each column of $X_{col}$ with the weights of the CONV Layer. The weights of the CONV layer are similarly stretched out into rows. For example, if there are 96 filters of size [11x11x3] this would give a matrix $W_row$ of size [96 x 363].<br>
    <br>
    The result of a convolution is now equivalent to performing one large matrix multiply `np.dot(W_row, X_col)`. In our example, the output of this operation would be [96 x 3025], giving the output of the dot product of each filter at each location.<br>
    <br>
    The result must finally be reshaped back to its proper output dimension [55x55x96].<br>
    <br>
    The downside is that it can use a lot of memory, since some values in the input volume are replicated multiple times in $X_{col}$. The benefit is that there are many very efficient implementations of Matrix Multiplication that we can take advantage of (e,g. BLAS API). 

!!! nt "**Note** (1x1 Convolution)"
    As an aside, several papers use 1x1 convolutions, as first investigated by [<u>Network in Network</u>](http://arxiv.org/abs/1312.4400).

## Pooling Layer
!!! df "**Definition** (Pooling Layer)"
    It is common to periodically insert a Pooling layer in-between successive Conv layers in a ConvNet architecture. The goal is: **To progressively reduce the spatial size of the representation, thus to reduce the amount of parameters and computation, and hence to control overfitting**. We use **MAX** operation to achieve these goals.<br>
    <br>
    The most common form is a pooling layer with filters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both width and height, discarding 75% of the activations. Every MAX operation would in this case be taking a max over 4 numbers (little 2x2 region in some depth slice). The depth dimension remains unchanged. More generally, the pooling layer: <br>
    <br>

    - Accepts a volume of size \( W_1 \times H_1 \times D_1 \)
    - Requires two hyperparameters:
    - their spatial extent \( F \),
    - the stride \( S \),
    - Produces a volume of size \( W_2 \times H_2 \times D_2 \) where:
    - \( W_2 = \left(\frac{W_1 - F}{S}\right) + 1 \)
    - \( H_2 = \left(\frac{H_1 - F}{S}\right) + 1 \)
    - \( D_2 = D_1 \)
    - Introduces zero parameters since it computes a fixed function of the input
    - For Pooling layers, it is not common to pad the input using zero-padding.

    In most cases, \( F = 3 \), \( S = 2 \) (also called overlapping pooling), or more commonly \( F = 2 \), \( S = 2 \). Pooling sizes with larger receptive fields are too destructive.

    <figure markdown="span">
    ![Image title](https://cs231n.github.io/assets/cnn/maxpool.jpeg){ width="600" }
    <figcaption>Pooling Layer</figcaption>
    </figure>

## Fully Connected (FC) Layer
Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset.

!!! im "**Important Note** (Converting FC Layers to CONV Layers)"
    Note that the neurons in both FC layers and CONV layers compute dot products, so their functional form is identical. Therefore, it turns out that it’s possible to convert between FC and CONV layers. Of these two conversions, the ability to convert an FC layer to a CONV layer is particularly useful in practice. For example, an FC layer with \( K = 4096 \) that is looking at some input volume of size \( 7 \times 7 \times 512 \) can be equivalently expressed as a CONV layer with \( F = 7, P = 0, S = 1, K = 4096 \). In other words, we are setting the filter size to be exactly the size of the input volume, and hence the output will simply be \( 1 \times 1 \times 4096 \) since only a single depth column “fits” across the input volume, giving identical result as the initial FC layer.

!!! eg "**Example** (FC-CONV Conversion in AlexNet)"
    Consider a ConvNet architecture that takes a 224x224x3 image, and then uses a series of CONV layers and POOL layers to reduce the image to an activations volume of size 7x7x512 (in an AlexNet architecture that we’ll see later, this is done by use of 5 pooling layers that downsample the input spatially by a factor of two each time, making the final spatial size 224/2/2/2/2/2 = 7). From there, an AlexNet uses two FC layers of size 4096 and finally the last FC layers with 1000 neurons that compute the class scores. We can convert each of these three FC layers to CONV layers as described above: <br>

    - Replace the first FC layer that looks at [7x7x512] volume with a CONV layer that uses filter size \( F = 7 \), giving output volume [1x1x4096].
    - Replace the second FC layer with a CONV layer that uses filter size \( F = 1 \), giving output volume [1x1x4096].
    - Replace the last FC layer similarly, with \( F = 1 \), giving final output [1x1x1000].

    Each of these conversions could in practice involve manipulating (e.g. reshaping) the weight matrix \( W \) in each FC layer into CONV layer filters. It turns out that this conversion allows us to “slide” the original ConvNet very efficiently across many spatial positions in a larger image, in a single forward pass.<br><br>
    

    For example, if 224x224 image gives a volume of size [7x7x512] - i.e. a reduction by 32, then forwarding an image of size 384x384 through the converted architecture would give the equivalent volume in size [12x12x512], since 384/32 = 12. Following through with the next 3 CONV layers that we just converted from FC layers would now give the final volume of size [6x6x1000], since (12 - 7)/1 + 1 = 6. Note that instead of a single vector of class scores of size [1x1x1000], we’re now getting an entire 6x6 array of class scores across the 384x384 image.<br><br>
    
    

    Evaluating the original ConvNet (with FC layers) independently across 224x224 crops of the 384x384 image in strides of 32 pixels gives an identical result to forwarding the converted ConvNet one time.<br><br>
    
    

    Naturally, forwarding the converted ConvNet a single time is much more efficient than iterating the original ConvNet over all those 36 locations, since the 36 evaluations share computation. This trick is often used in practice to get better performance, where for example, it is common to resize an image to make it bigger, use a converted ConvNet to evaluate the class scores at many spatial positions and then average the class scores.