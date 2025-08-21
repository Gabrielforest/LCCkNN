# kKNN: Adaptive k-Nearest Neighbor Classifier

## Project Overview

The **LCCkNN** R package implements an adaptive k-nearest neighbor (k-NN) classifier based on local curvature estimation. Unlike the traditional k-NN algorithm, which uses a fixed number of neighbors ($k$) for all data points, the kK-NN algorithm dynamically adjusts the neighborhood size for each sample.

The core idea is that data points with low curvature could have larger neighborhoods, as the tangent space approximates well the underlying data shape. Conversely, points with high curvature could have smaller neighborhoods, because the tangent space is a loose approximation. This adaptive strategy is capable of avoiding both underfitting and overfitting, and improves classification performance, especially when dealing with a limited number of training samples. 
The algorithm's key components include:

  * **Curvature Estimation**: The local Gaussian curvature is estimated by approximating the local shape operator in terms of the local covariance matrix and the local Hessian matrix. This approximation for the local shape operator of the data manifold improves the robustness to noise and outliers.
  * **Adaptive Neighborhood Sizing**: The curvatures are quantized into ten different scores. Based on these scores, the adaptive neighborhood adjustment is performed by pruning the edges of the k-NN graph, reducing the neighborhood size for high-curvature points and retaining a larger neighborhood for low-curvature points.
  * **Improved Performance**: Results on many real-world datasets indicate that the kK-NN algorithm yields superior balanced accuracy compared to the established k-NN method and another adaptive k-NN algorithm. The results consistently show that the kK-NN classifier is superior to the regular k-NN and also the competing adaptive k-NN algorithm.


## Key Feature: Dual Quantization Support
This package offers two distinct quantization methods for adaptive neighborhood sizing:


Paper Method: The default approach, which quantizes curvatures into ten different scores (from 0 to 9), as described in the original paper.

Log2n Method: A slightly modified approach, also developed by the same author, that quantizes curvatures into k scores, where k is the number of neighbors, which is calculated as k=
log2n.

This dual-method support allows users to test and compare both quantization strategies within a single package.

## 📦 Installation

You can install the `LCCkNN` package directly from GitHub using `devtools`.

```r
devtools::install_github("gabrielforest/LCCkNN")
```

## Usage Example

Here is a quick example of how to use the `kKNN` function on the built-in `iris` dataset.

```r
# Load necessary libraries
library(caret)

# Load and prepare data (e.g., the Iris dataset)
data_iris <- iris
data <- as.matrix(data_iris[, 1:4])
target <- as.integer(data_iris$Species)

# Standardize the data
data <- scale(data)

# Split data into training and testing sets
set.seed(42)
train_index <- caret::createDataPartition(target, p = 0.5, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]
train_labels <- target[train_index]

# Determine initial k value as log2(n)
initial_k <- round(log2(nrow(train_data)))
if (initial_k %% 2 == 0) {
   initial_k <- initial_k + 1
}

# Run the kK-NN classifier using the default quantization method ('paper')
predictions_paper <- LCCkNN::kKNN(
   train = train_data,
   test = test_data,
   train_target = train_labels,
   k = initial_k
)

# Run the kK-NN classifier using the 'log2n' quantization method
predictions_log2n <- LCCkNN::kKNN(
   train = train_data,
   test = test_data,
   train_target = train_labels,
   k = initial_k,
   quantize_method = 'log2n'
)

# Evaluate the results (e.g., calculate balanced accuracy)
test_labels <- target[-train_index]
bal_acc_paper <- LCCkNN::balanced_accuracy_score(test_labels, predictions_paper)
bal_acc_log2n <- LCCkNN::balanced_accuracy_score(test_labels, predictions_log2n)
cat("Balanced Accuracy (paper Method):", bal_acc_paper, "\n")
cat("Balanced Accuracy (log2n Method):", bal_acc_log2n, "\n")
```

## Citation

This package is an R implementation of the algorithm proposed in the following research paper. For more details on the methodology, please refer to the original article:

> Levada, A. L. M., Nielsen, F., & Haddad, M. F. C. (2024). **ADAPTIVE k-NEAREST NEIGHBOR CLASSIFIER BASED ON THE LOCAL ESTIMATION OF THE SHAPE OPERATOR**. *arXiv preprint arXiv:2409.05084*.
>
> **[Read the full paper here](https://arxiv.org/pdf/2409.05084)**
