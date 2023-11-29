# Understanding Logistic Regression

Logistic regression is a statistical method used for binary classification, which is the task of classifying elements into two distinct groups. 
This algorithm is particularly useful for answering questions that have a "yes" or "no" answer, such as determining if an email is spam (yes/no) or if a tumor is malignant (yes/no).

## The Logistic Regression Model

In logistic regression, we calculate the probability that a given input X belongs to the default class y (usually the class labeled as 1). This probability is denoted as p(y | X; θ), where θ represents the parameters of our model.
To predict this probability, we use the sigmoid function, denoted as h(x) in its simplest form, which outputs a value between 0 and 1. This output can be interpreted as the probability that X belongs to class 1.
Optimal Prediction Function
When discussing the optimal prediction function, we're looking at how likely our data is under the current model parameters. We represent this likelihood as h(x)^y * (1 - h(x))^(1 - y). It means:
If the true class y is 1, the probability of the model predicting the true class is h(x).
Conversely, if the true class y is 0, the probability of the model predicting the true class is 1 - h(x).
The model is most accurate when these probabilities are high for all observations in our dataset.
The Cost Function
The cost function, also known as the loss function, is a measure of how well our model is doing. For logistic regression, we typically use the log loss function, which compares the predicted probabilities against the actual class labels. It's given by the formula:
J(θ) = -1/m * Σ [y(i) * log(hθ(x(i))) + (1 - y(i)) * log(1 - hθ(x(i)))]

Where:
m is the number of observations in our dataset.
y(i) is the actual class label of the i-th observation.
hθ(x(i)) is the model's predicted probability that the i-th observation belongs to class 1.
Our goal during training is to adjust the parameters θ to minimize J(θ).
Gradient Descent
Gradient descent is an optimization algorithm used to minimize the cost function. It works by iteratively adjusting the parameters θ in the opposite direction of the gradient of the cost function.
The basic steps are:
Calculate the gradient: Determine the direction in which the cost function is increasing.
Update the parameters: Move the parameters θ in the opposite direction of the gradient.
Repeat: Keep updating the parameters until the cost function stops decreasing (or decreases very slowly).
Through this process, we find the values of θ that result in the lowest possible error, which corresponds to the most accurate predictions our model can make given the data.
