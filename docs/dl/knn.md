# k - Nearest Neighbor Classifier (kNN)

## Nearest Neighbor Classifier (NN)
Nearest Neighbor Classifier is very rarely used in practice, but it will allow us to get an idea about the basic approach to an image classification problem.

!!! im "**Important Note** (CIFAR-10 Dataset)"
    One popular toy image classification dataset is the [CIFAR-10 dataset](https://cs231n.github.io/classification/). This dataset consists of 60,000 tiny images that are 32 pixels high and wide. Each image is labeled with one of 10 classes (for example “airplane, automobile, bird, etc”). These 60,000 images are partitioned into a training set of 50,000 images and a test set of 10,000 images.

    <figure markdown="span">
    ![Image title](https://cs231n.github.io/assets/nn.jpg){ width="800" }
    <figcaption>CIFAR-10 Dataset</figcaption>
    </figure>

Before introducing algorithms to solve an image classification problem, we need to first define the difference between two images. One of the simplest possibilities is to compare the images pixel by pixel and add up all the differences. 

!!! df "**Definition** (L1 Distance)"
    Given two images and representing them as vectors \(I_1, I_2\), a reasonable choice for comparing them might be the _L1 distance_:

    \[d_1(I_1, I_2) = \sum_p |I_1^p - I_2^p|\]

    Here is the procedure visualized:

    <figure markdown="span">
    ![Image title](https://cs231n.github.io/assets/nneg.jpeg){ width="600" }
    <figcaption></figcaption>
    </figure>

There are many other ways of computing distances between vectors. Another common choice could be to instead use the **L2 distance**, which has the geometric interpretation of computing the Euclidean distance between two vectors. 

!!! df "**Definition** (L2 Distance)"
    The _L2 distance_ takes the form:

    \[d_2(I_1, I_2) = \sqrt{\sum_p (I_1^p - I_2^p)^2}\]

!!! df "**Definition** (Nearest Neighbor Classifier)"
    Suppose now that we are given the CIFAR-10 training set of 50,000 images (5,000 images for every one of the labels), and we wish to label the remaining 10,000. The nearest neighbor classifier will take a test image, compare it to every single one of the training images, and predict the label of the closest training image. In the image "CIFAR-10 Dataset" you can see an example result of such a procedure for 10 example test images.

!!! nt "**Note** (Performance of Nearest Neighbor Classifier)"
    If you ran the Nearest Neighbor classifier on CIFAR-10 with L2 distance, you would obtain 35.4% accuracy (slightly lower than our L1 distance result - 38.6%).

## k - Nearest Neighbor Classifier (kNN)
You may have noticed that it is strange to only use the label of the nearest image when we wish to make a prediction. Indeed, it is almost always the case that one can do better by using what’s called a **k-Nearest Neighbor Classifier**.

!!! df "**Definition** (k - Nearest Neighbor Classifier)"
    instead of finding the single closest image in the training set, we will find the top k closest images, and have them vote on the label of the test image. In particular, when k = 1, we recover the Nearest Neighbor classifier. Intuitively, higher values of k have a smoothing effect that makes the classifier more resistant to outliers:

    <figure markdown="span">
    ![Image title](https://cs231n.github.io/assets/knn.jpeg){ width="800" }
    <figcaption>k - Nearest Neighbor Classifier</figcaption>
    </figure>

!!! df "**Definition** (Validation Set)"
    There is a correct way of tuning the hyperparameters and it does not touch the test set at all. The idea is to split our training set in two: a slightly smaller training set, and what we call a validation set. Using CIFAR-10 as an example, we could for example use 49,000 of the training images for training, and leave 1,000 aside for validation. This validation set is essentially used as a fake test set to tune the hyper-parameters. <br>

    Split your training set into training set and a validation set. Use validation set to tune all hyperparameters. At the end run a single time on the test set and report performance.

!!! df "**Definition** (Cross-Validation)"
    In cases where the size of your training data (and therefore also the validation data) might be small, people sometimes use a more sophisticated technique for hyperparameter tuning called cross-validation. Working with our previous example, the idea is that instead of arbitrarily picking the first 1000 datapoints to be the validation set and rest training set, you can get a better and less noisy estimate of how well a certain value of k works by iterating over different validation sets and averaging the performance across these. For example, in 5-fold cross-validation, we would split the training data into 5 equal folds, use 4 of them for training, and 1 for validation. We would then iterate over which fold is the validation fold, evaluate the performance, and finally average the performance across the different folds.

!!! im "**Important Note** (In Practice)"
    In practice, people prefer to avoid cross-validation in favor of having a single validation split, since cross-validation can be computationally expensive. The splits people tend to use is between 50%-90% of the training data for training and rest for validation. However, this depends on multiple factors: For example if the number of hyperparameters is large you may prefer to use bigger validation splits. If the number of examples in the validation set is small (perhaps only a few hundred or so), it is safer to use cross-validation. Typical number of folds you can see in practice would be 3-fold, 5-fold or 10-fold cross-validation.

    <figure markdown="span">
    ![Image title](https://cs231n.github.io/assets/crossval.jpeg){ width="800" }
    <figcaption>Common Data Splits</figcaption>
    </figure>
    
!!! im "**Important Note** (Pros and Cons of NN)"
    The nearest neighbor classifier takes no time to train, since all that is required is to store and possibly index the training data. However, we pay that computational cost at test time, since classifying a test example requires a comparison to every single training example. 

!!! im "**Important Note** (Problems of NN)"
    Images are high-dimensional objects (i.e. they often contain many pixels), and distances over high-dimensional spaces can be very counter-intuitive. The image below illustrates the point that the pixel-based L2 similarities we developed above are very different from perceptual similarities:

    <figure markdown="span">
    ![Image title](https://cs231n.github.io/assets/samenorm.png){ width="600" }
    </figure>

    [Here](https://cs231n.github.io/assets/pixels_embed_cifar10_big.jpg) is one more visualization to convince you that using pixel differences to compare images is inadequate. We can use a visualization technique called t-SNE to take the CIFAR-10 images and embed them in two dimensions so that their (local) pairwise distances are best preserved. In this visualization, images that are shown nearby are considered to be very near according to the L2 pixelwise distance we developed above:

    <figure markdown="span">
    ![Image title](https://cs231n.github.io/assets/pixels_embed_cifar10.jpg){ width="800" }
    <figcaption></figcaption>
    </figure>
    
    In particular, note that images that are nearby each other are much more a function of the general color distribution of the images, or the type of background rather than their semantic identity.