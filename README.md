# Overview of the Original Paper

The paper "Sharpness-Aware Minimization for Efficiently Improving Generalization" proposes an optimization method designed to enhance the generalization capabilities of neural networks. The technique, known as Sharpness-Aware Minimization (SAM), is groundbreaking as it shifts the focus from traditional loss minimization to also including the smoothness of the loss landscape. The paper describes that smoother loss landscapes correlate with better generalization in unseen data. This optimization strategy has been validated in their paper and is shown to improve performance across various benchmarks, making it a significant contribution to the field of machine learning

# Scope of Evaluation

Our evaluation aims to replicate the findings of the original paper using the MNIST fashion dataset and CIFAR-10 dataset. The evaluation is based on the following criteria:  model accuracy, generalization to test datasets, and comparison between SAM and SGD across different epoch settings. Our replication seeks to confirm the findings of the original paper that shows SAM’s robustness over SGD.

# Dataset

For our replication study, we chose the CIFAR-10 and MNIST fashion datasets, both widely recognized benchmarks in the machine learning field. CIFAR-10 is known for its complexity and diversity in image classification tasks, and the MNIST fashion dataset provides a set of grayscale images of apparel that provide their own challenges. The choice of these datasets is relevant as they are standard for evaluating the generalization improvements claimed by SAM and are different enough from each other to provide a more thorough review.

# Data Pre-processing

To ensure that training was effective, we implemented a series of pre-processing steps. For both datasets, we normalized the images to have a consistent mean and standard deviation, aiding in model convergence and performance. We applied random crops and horizontal flips to introduce variability, in order to simulate data augmentation techniques that improve model robustness. Additionally, we utilized a technique known as Cutout as a form of regularization, randomly masking out sections of input images during training to further push the model towards better generalization.

—--------------------------

VIRAJ

—--------------------------

# Experiment SAM vs. SGD\
The experiment was designed to compare the performance of the optimization algorithms Sharpness-Aware Minimization (SAM) and Stochastic Gradient Descent (SGD) on training a wide residual network (Wide-res-net). The model was trained on two datasets: CIFAR-10 and MNIST Fashion. The training was carried out over two different lengths: 100 epochs and 200 epochs. Each configuration was run three times to ensure the reliability of the results.

The hypothesis is that SAM will perform better than SGD because it is better at generalizing.

|              |        |       |       |         |         |
| ------------ | -----: | ----- | ----- | ------- | ------- |
|              | Epochs | Sam   | SGD   | Sam     | SGD     |
| Wide-res-net |    100 | 96.58 % | 96.29 % | 95.70 % | 95.67 % |
|              |    100 | 96.47 % | 96.43 % | 95.54 % | 95.91 % |
|              |    100 | 96.18 % | 96.38 % | 95.46 % | 95.79 % |
| Average:     |        | 96.41 % | 96.37 % | 95.57 % | 95.79 % |
|              |    200 | 96.69 % | 96.86 % | 95.77 % | 95.90 % |
|              |    200 | 96.96 % | 96.83 % | 96.04 % | 95.94 % |
|              |    200 | 96.98 % | 96.87 % | 95.77 % | 95.92 % |
| Average:     |        | 96.88 % | 96.85 % | 95.86 % | 95.92 % |

The results indicate that the performance of SAM and SGD is roughly equivalent, which is unexpected given the findings from the original research paper, where SAM was reported to outperform SGD. This discrepancy could be attributed to several factors: The architecture of the Wide-res-net used in our experiments, although intended to mimic that of the research paper, might have differences in layer configurations, activation functions, or other architectural details that affect the performance of the optimizers.

Experiment SimpleNN model SGD vs SAM\
The first experiment did not give the results we expected, so we decided to do a simpler experiment with a smaller dataset and model. A smaller experiment is easier to understand and conduct so this seemed like a good option. Instead of the big wideResNet model, we used a simple 3-layer fully connected model to learn to classify the classic MNIST digits dataset.

This experiment evaluates the performance of Stochastic Gradient Descent (SGD) and Sharpness-Aware Minimization (SAM) on a simple neural network model—a fully connected, two-layer neural network—using the MNIST digits dataset.

# Methodology

Hyperparameter Optimization:

SAM: We optimized both the learning rate and the rho parameter through a grid search. The learning rates tested were \[0.01, 0.05, 0.1, 0.2, 0.3] and the rho values were \[0.01, 0.05, 0.1].

SGD: Optimization focused solely on the learning rate, testing the same range of values as SAM.

Experimental Setup

Upon establishing the optimal hyperparameters (learning rate=0.05), the simple neural network model was trained using each optimizer configuration five times, with each run lasting 100 epochs. This extensive testing ensures the reliability of our findings.

The results are:

|          |        |        |                 |
| -------- | -----: | ------ | --------------- |
|          |        | MNIST  |                 |
|          | Epochs | SAM    | SGD             |
| SimpleNN |    100 | 5 runs | 5 runs          |
|          |        | 98.79  | 98.73           |
|          |        | 98.79  | 98.73           |
|          |        | 98.79  | 98.73           |
|          |        | 98.79  | 98.73           |
|          |        | 98.79  | 98.73           |

 

The results show that SAM performs better than SGD, but only slightly.

# Discussion

Sadly, once again the results don't tell us much. SAM came out ahead by a very tiny margin, but nothing significant. Interestingly for both SAM and SGD over the 5 runs there was 0 variance, indicating a very easy to find minimum. This might explain the lack of difference between the two. In simpler models, which typically have smoother loss landscapes, the benefits of SAM might be less noticeable. SAM is made to find flatter minima in the loss landscape, which are associated with better generalization on unseen data. However, because simpler models inherently have fewer sharp minima, the landscape is already relatively flat. This means that the improvement SAM offers could be minimal compared to SGD.

# Conclusion

In conclusion, SAM is an interesting variant on optimizers that minimizes the sharpness of the loss landscape while simultaneously minimizing the loss. This causes the resulting network to generalize better to data not in the training set at the cost of an extra backpropagation at every step. We performed two experiments, the first was a reproduction of a part of the experiments performed in the paper. We saw quite different results than reported in the original paper. There was no significant difference between SAM and SGD. We hypothesized some possible reasons that could explain this difference and performed another experiment. This experiment was simpler than the first but still, the same results were found. From this, we can conclude that when optimizing a wideResNet on both the Cifar-10 and fashion MNIST dataset, there is no significant advantage to using a SAM optimizer over the pytorch SGD optimizer. Lastly, we found that a simple network architecture also does not give SAM a sizable advantage over SGD.

# Future work:

\- Try more complex data sets like Cifar-100 and Caltech101

\- Over the years some variants of SAM have been researched. The experiments can be repeated using those.

\- Sam can be compared against more modern optimizers, like Adam for instance.
