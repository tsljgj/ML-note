# Unsupervised Learning

Unlike supervised learning, unsupervised learning problems do not have labels over items.

## The k-Means Clustering Algorithm

!!! df "**Definition** (k-Means Clustering Algorithm)"
    In the clustering problem, we are given a training set $\{x^{(1)}, \ldots, x^{(n)}\}$, and want to group the data into a few cohesive “clusters.” Here, $x^{(i)} \in \mathbb{R}^d$ as usual; but no labels $y^{(i)}$ are given. So, this is an unsupervised learning problem.

    The _k-means clustering algorithm_ is as follows:

    1. Initialize **cluster centroids** $\mu_1, \mu_2, \ldots, \mu_k \in \mathbb{R}^d$ randomly.

    2. Repeat until convergence: `{` <br>

        For every $i$, set

        $$
        c^{(i)} := \arg \min_j \|x^{(i)} - \mu_j\|^2.
        $$

        For each $j$, set

        $$
        \mu_j := \frac{\sum_{i=1}^n 1\{c^{(i)} = j\} x^{(i)}}{\sum_{i=1}^n 1\{c^{(i)} = j\}}.
        $$

        where $1\{c^{(i)} = j\}$ is an indicator function that is 1 if the $i$-th data point is assigned to cluster $j$, and 0 otherwise.
    &nbsp;&nbsp;&nbsp;&nbsp;`}`

    In the algorithm above, $k$ (a parameter of the algorithm) is the number of clusters we want to find; and the cluster centroids $\mu_j$ represent our current guesses for the positions of the centers of the clusters. To initialize the cluster centroids (in step 1 of the algorithm above), we could choose $k$ training examples randomly, and set the cluster centroids to be equal to the values of these $k$ examples. (Other initialization methods are also possible.)

    The inner-loop of the algorithm repeatedly carries out two steps: (i) “Assigning” each training example $x^{(i)}$ to the closest cluster centroid $\mu_j$, and (ii) Moving each cluster centroid $\mu_j$ to the mean of the points assigned to it. 

    <figure markdown="span">
    ![Image title](https://stanford.edu/~shervine/teaching/cs-229/illustrations/k-means-en.png?9925605d814ddadebcae2ae4754ab0a4){ width="800" }
    <figcaption>k-means Clustering</figcaption>
    </figure>

Is the k-means algorithm guaranteed to converge? Yes it is, in a certain sense. In particular, let us define the **distortion function**.

!!! df "**Definition** (Distortion Function)"
    Define the distortion function to be:

    $$
    J(c, \mu) = \sum_{i=1}^n \|x^{(i)} - \mu_{c(i)}\|^2
    $$

    $J$ measures the sum of squared distances between each training example $x^{(i)}$ and the cluster centroid $\mu_{c(i)}$ to which it has been assigned. It can be shown that k-means is exactly **coordinate descent** on $J$. Specifically, the inner-loop of k-means repeatedly minimizes $J$ with respect to $c$ while holding $\mu$ fixed, and then minimizes $J$ with respect to $\mu$ while holding $c$ fixed. Thus, $J$ must monotonically decrease, and the value of $J$ must converge. Usually, this implies that $c$ and $\mu$ will converge too.

!!! df "**Definition** (Coordinate Descent)"
    Coordinate descent is an optimization algorithm used to minimize a function by iteratively selecting one coordinate (or variable) at a time and optimizing the objective function with respect to that coordinate while keeping the other coordinates fixed.<br><br>

    This is how it works:

    1. **Initialization**: Start with an initial guess for the variables \(\mathbf{x} = (x_1, x_2, \ldots, x_n)\).

    2. **Iterative Optimization**:
        - For each coordinate \(i\), optimize the objective function \(f(\mathbf{x})\) with respect to \(x_i\) while keeping all other coordinates fixed.
        - Update \(x_i\) to the value that minimizes \(f(\mathbf{x})\) in this step.

    3. **Repeat**: Continue iterating over all coordinates until convergence, i.e., until the change in the objective function or the variables is below a predefined threshold.

!!! nt "**Note** (Limitation of k-means)"
    The distortion function $J$ is a non-convex function, and so coordinate
    descent on $J$ is not guaranteed to converge to the global minimum. In other
    words, k-means can be susceptible to local optima. Very often k-means will
    work fine and come up with very good clusterings despite this. But if you
    are worried about getting stuck in bad local minima, one common thing to
    do is run k-means many times (using different random initial values for the
    cluster centroids $\mu_{j}$ ). Then, out of all the different clusterings found, pick
    the one that gives the lowest distortion $J(c, \mu)$.

- [ ] Mixture of Gaussians (GMM)
- [ ] EM-Algorithm
- [ ] Factor Analysis
- [ ] Principal Components Analysis (PCA)
- [ ] Independent Components Analysis (ICA)
    
    