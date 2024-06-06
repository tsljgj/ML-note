# Linear Regression: Softmax

In addition to the house price prediction problem, we are interested in another kind of linear regression problem: classification. Instead of asking "how much," we now ask "which."

## Softmax Regression
!!! def "**Definition** (Classification Problem)"
    There are two kinds of _classification_ problem:<br>
    &nbsp;&nbsp;&nbsp;&nbsp;1. Hard assignments of examples to categories<br>
    &nbsp;&nbsp;&nbsp;&nbsp;2. Assess the probaility that each category applies

??? eg "**Example** (Classification Problem Example)"
    Consider a $2 \times 2$ image. Let $x_1, x_2, x_3, x_4$ denote the scalar value of each pixel. These are the four features in our model. Furthermore, assume that each image belongs to one of the categories "cat," "chicken," or "dog." We want to determine the class to which a given image belongs.

!!! def "**Definition** (One-hot Encoding)" 
    For a sample in an $n$-category classification problem, the _hot-encoding_ of that sample is a vector with $n$ components, and the only component corresponding to the sample's category is set to 1 and all other components are set to 0.

??? eg "**Example** (One-hot Encoding Example)"
    In the previous example, $n = 3$, so "cat" can be encoded as $(1, 0, 0)$, "chicken" ... $(0, 1, 0)$, "dog" ... $(0, 0, 1)$.

!!! note "**Note** (Ordered-Category Encoding)"
    If categories had some natural ordering among them, we can encode them in a much more intuitive way. For example, say we want to predict $\{\text{baby, toddler, adolescent}\}$, then it might make sense to cast this as an ordinal regression problem and keep the labels in this format: $\text{label} \in \{1, 2, 3\}$

!!! def "**Model** (Linear Model for Classification)"
    Consider a network with one output layer and one input layer. For classification problems, #nodes in the output layer $=$ #category.
    <figure markdown="span">![Image title](https://d2l.ai/_images/softmaxreg.svg){ width="500" }<figcaption>Linear Model for Classification Problem </figcaption></figure> 
    Let $\mathbf{o}$ denotes the output (a vector) of neural network, we have:
    <br><br>
    <center>$\mathbf{o} = \mathbf{W}\mathbf{x} + \mathbf{b}$</center>
    <br>

At this point, we can, assuming a suitable loss function, try to minimize the difference between $\mathbf{o}$ and the hot-encoding of the sample. In fact, this works surprisingly well. However, we need to consider two drawbacks of this method:<br>
&nbsp;&nbsp;&nbsp;&nbsp;1. $\sum \mathbf{o}_i \neq 1$<br>
&nbsp;&nbsp;&nbsp;&nbsp;2. $\mathbf{o}_i$ may be negative<br>
To address these issues, we use _softmax_.

!!! def "**Definition** (Softmax Function)"
    <center>$\hat{\mathbf{y}} = \text{softmax}(\mathbf{o})$ where $\hat{y}_i = \frac{\text{exp}(\mathbf{o}_i)}{\sum_{j}\text{exp}(\mathbf{o}_j)}.$</center>
