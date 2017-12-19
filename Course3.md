# Orthogonalization

Orthogonalization or orthogonality is a system design property that assures that modifying an instruction or a component of an algorithm will not create or propagate side effects to other components of the system. In simple words: what to tune in order to achieve what effect.

### Chain of Assumption in ML

When a supervised learning system is design, these are the 4 assumptions that needs to be true and orthogonal.
1. Fit training set well in cost function
    - If it doesn’t fit well, the use of a *bigger neural network* or switching to a *better optimization algorithm* like Adam might help.

2. Fit development set well on cost function
    - If it doesn’t fit well, *regularization* or using *bigger training set* might help. 

3. Fit test set well on cost function
    - If it doesn’t fit well, the use of a *bigger development set* might help 

4. Performs well in real world
    - If it doesn’t perform well, the *development test set is not set correctly* or *the cost function is not evaluating the right thing*.

Tend not to use early stopping. It makes you fit less well in the training set, and it affects the performance in Training and Dev test at the same time. And it's not orthogonal.

# Set up your Goal

### Single number evaluation metric 
  - precision vs. recall
  - use F1 score (average of precision and recall)
  - $F1 = \frac {2} {\frac {1} {P} + \frac {1} {R}}$ - Harmonic mean
  - Dev set + single real number evaluation metric speeds up iterating

- Satisfying and Optimizing metric
    - combine the accuracy and running time
    - Or maximize accuracy but subject to <= 100 ms
    - General rule:

$$
   N_{metric}=\begin{cases}
     1, & \text{Optimizing metric}\\
     N_{metric}-1, & \text{Satisficing metric}
   \end{cases}
$$


### Train/Dev/Test distribution

- Make sure Dev and Test come from the same distribution
- Randomly shuffle data into Dev and Test set
- Choose dev and test set to reflect data you expect to get in the future

### Size of dev and test sets

- Old way of splitting data:
    - Train/test: 70% / 30%
    - Train / Dev / Test: 60% / 20% / 20%
    - OK when you have data size 1000-10000 examples

- Modern way:
    - Train / Dev / Test: 98% / 1% / 1%
    - Data size is 1,000,000 training examples

- Test set - big enough to give high confidence in the overall performance

- Test set helps evaluate the performance of the final classifier which could be less 30% of the whole data set.

- The development set has to be big enough to evaluate different ideas.

### When to change Dev/Test set and metrics

$$ Error = \frac {1} {m_{dev}} \sum_{i=1}^{m_{dev}} L(yhat^{(i)} \neq y^{(i)})$$

This function counts up the number of misclassified examples. The problem with this evaluation metric is that it treats pornographic vs non-pornographic images equally.

One way to change this evaluation metric is to add the weight term 𝑤(𝑖)

$$
   w^{(i)}=\begin{cases}
     1  & \text{if x(i) is pornografic}\\
     10 & \text{if x(i) is non-pornografic}
   \end{cases}
$$

The function becomes:

$$ Error = \frac {1} {\sum {w^{(i)}}} \sum_{i=1}^{m_{dev}} w^{(i)}L(yhat^{(i)} \neq y^{(i)})$$

Guideline
1. Define correctly an evaluation metric that helps better rank order classifiers
2. Optimize the evaluation metric


# Comparing with human level performance

- Bayes Optimal Error: Best optimial error (cannot be surpassed)

- If ML is worse than human, you can:
    - Get labelled data from human.
    - Gain insight from manual error analysis.
    - Better analysis of bias and variance.

### Avoidable bias

- If human level is 1%, but training error is 8% and dev error is 10%, then focus on reducing bias. (bigger network?)
- If human level is 7.5% error, training error 8% and dev error is 10%, focus on reducing variance. (Regularization?)
- Avoidable Bias: difference between bayes error and training error.

### Understanding Human Level Performance

- Human level error as a proxy for Bayer error
- Choose variance reduction technique or bias reduction technique according to the difference between human level performance and training performance/testing performance
- If the difference between human-level error and the training error is bigger than the difference between the training error and the development error. The focus should be on bias reduction technique
- If the difference between training error and the development error is bigger than the difference between the human-level error and the training error. The focus should be on variance reduction technique


### Surpassing human level performance

There are many problems where machine learning significantly surpasses human-level performance,
especially with structured data:
- Online advertising
- Product recommendations
- Logistics (predicting transit time)
- Loan approvals

### Improving your model
The two fundamental assumptions of supervised learning:
- You can fit training set really well. (Low avoidable bias)
- The training set performance generalized pretty well to dev/test set

Human level

$$
\text{Avoidable bias} \begin{cases}
     \text{Train bigger model}\\
     \text{Train longer, better optimization algorithms (Momentum, RMSprop, Adam)}\\
     \text{Neural Networks architecture/hyperparameters search}
   \end{cases}
$$

Training error

$$
\text{Variance} \begin{cases}
     \text{More data}\\
     \text{Regularization (L2, dropout, data augmentation)}\\
     \text{Neural Networks architecture/hyperparameters search}
   \end{cases}
$$

Development error


# Error Analysis

### Carrying out error analysis

- Ceiling of a solution. (If you solve this problem perfectly, how much improvement do you got?)
- find a set of mislabelled example of the dev set, look at those example for either false positive and false negative and count up errors for each category.


### Clearning up incorrectly labelled data

- DL algorithms are robust to random errors

- Look at three numbers:

  1. Overall dev set error.
  2. Errors due to incorrect labels.
  3. Errors due to all other causes.

- Apply same process to your dev ans test sets (make sure they come from the same distribution)

- Consider examining examples your algorithm got right


### Build your first system quickly and iterate

1. Set up development/ test set and metrics
    - Set up a target
2. Build an initial system quickly
    - Train training set quickly: Fit the parameters
    - Development set: Tune the parameters
    - Test set: Assess the performance
3. Use Bias/Variance analysis & Error analysis to prioritize next steps


# Mismatched training and dev/test set

What if training data comes from different data?
- Option One: Randomly shuffle data from different distribution, and then put them into Train/Dev/Test sets. 
    - Adventage: data comes from the same distribution
    - Disadventage: end goal might be only one of those distributions instead of all of them
- Option Two: Setting the dev and test set come only from target distribution. (This approach is better)
    - Advantage: Aiming the target where you wanted to be
    - Disadvantage: training set has different distribution

### Bias and Variance with mismatched data distribution

When the training set is from a different distribution than the development and test sets the method to analyze bias and variance changes
  - If training and dev comes from the same distribution, then our classifier might have a high bias problem
  - When the training set, development and test sets distributions are different, we define a new subset called training-development set. This new subset has the same distribution as the training set, but it is not used for training the neural network.

Principal:

- Human level error.
- Training set error.
- Training-dev set error.
- Dev set error.
- Look at the four quantity above to see if it's bias/variance/data mismatch problem.

General formulation

Bayes error

    - Avoidable Bias

Training set error
    - Variance

Development - Training set error

    - Data mismatch

Development set error

    - Degree of overfitting to the development set

Test set error


### Address data mismatch problem

- Carry out manual error analysis to try to understand difference between training and dev/test set. Development should never be done on test set to avoid overfitting.

- Make training data or collect data similar to development and test sets. To make the training data more similar to your development set, you can use is artificial data synthesis. However, it is possible that if you might be accidentally simulating data only from a tiny subset of the space of all possible examples.


# Learning from Multiple Tasks

### Transfer learning
 Transfer learning refers to using the neural network knowledge for another application.

- Delete last layer of neural network
- Delete weights feeding into the last output layer of the neural network
- Create a new set of randomly initialized weights for the last layer only
- New data set (𝑥, 𝑦)
- Retrain the network on the new data set

If you have a small training set then retrain only the last layer.

When transfer learning makes sense:

  - Tasks A and B habe same input x (e.g. both images)
  - You hava a lot more data for Task A then Task B
  - low level features form A could be helpful for learning B

### Multi-task Learning

In multi-task learning, you start off simultaneously, trying to have one neural network do several things at the same time. And then each of these task helps hopefully all of the other task.

Instead of using softmax loss function, who predict one label for an example, we can summing up the logistic loss:

$$ Error = \frac {1} {m} \sum_{i=1}^{m} \sum_{j=1}^{4} L(yhat_{j}^{(i)}, y_{j}^{(i)}) $$

Loss: $yhat^{(i)}$ (4,1) Vector

$$L = -y_{j}^{(i)} log(yhat_{j}^{(i)}) - (1 - y_{j}^{(i)})log(1-yhat_{j}^{(i)})$$

When multi-task learning makes sense:

- Training on a set of tasks that share lower level features.
- Amount of data you have for each task is quite similar.
- Can train a big enough network to do well on all tasks.

# End to End deep learning

End-to-end deep learning is the simplification of a processing or learning systems into one neural network.

Bypass preprocessing stages and really simplifies the design of the system. But it works only if you have a lot of data.

### Whether to use end-to-end deep learning

- Pros:
  - Let the data speak instead of having to enforcing human pre-conception.
  - Less hand-designing of components is needed.
- Cons:
  - Need a large number of data.
  - Excludes potentially useful hand-designed component

- Key question:
  - Do you have enough data to learn a function of the complexity needed to map x to y.