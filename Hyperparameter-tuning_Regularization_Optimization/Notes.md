# Regularization

Regularization helps to prevent overfitting. Another way to address high variance is to get more data.
<!---
$$J = -\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small  y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)}$$
--->

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/01.gif "")

<!---
$$ J(w,b)=\frac{1}{m} \sum_{i=1}^{m} L(yhat^{(i)}, y^{(i)}) + \underbrace{\frac{\lambda}{2m} ||w||_2^2} _\text {cross-entropy cost}$$
-->

L_2 regularization:

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/02.gif "")
<!---
$$||w||_2^2 = \sum_{j=1}^{n_x} w_j^2 = w^Tw$$
-->


![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/02b.gif "") is regularization parameter

- Regularization of Logistic Regression: 
    - Often regularize w instead of b, w is a vector while b is a scalar. Most parameters are in w. Omiting b has no influence.
    - If we use L1 norm regularization, w would become more sparse. (May help compressing the model, but the difference is not obvious)
    - L2 norm regularization is much more often. 

- Regularization on NN:
    - Frobenuius norm: sum of the square of elements in weight matrix.
    - With L2-norm, Weight matrix is becoming smaller, L2-norm is also called "weight decay"
  
- Why regularization reduces overfitting?
    - Less weight -> 'zero' out the impact of hidden units -> simpler network. 
    - tanh activation function -> (less weight)-(smaller z) -> linear region of tanh -> every layer would be roughly linear. 

- Dropout
    - Inverted Dropout(Remember to divide keep_prob to ensure the expected value to be the same)
    - Keep_prob can be different for different layer
    - Intuition1: Knocking out neurons -> smaller network -> regularing effect
    - Intuition2: Each neuron can't rely on only one feature, and much more motivated to spread out weight -> shrinking weights.
    - Dropout doesn't always generalize to other discipline. Although in CV. it's almost default. Remember, Dropout is for regularization!
    - Dropout makes cost function less well defined!
    
            When implement dropout for backprop,remember to scale the derivative by keep_prob, just as you did in forward-prop for the activation values. 
    
- Other techniques
    - Data Augmentation. (Inexpensive way)
    - Early Stopping (According to the error rate of dev set)
    
# Optimization

- Normalizing input features -> affect your cost function shape(more round and easier to optimize) -> speed up gradient descent.

- If the elements in weight matrix is a bit larger than one, in a very deep network, the activation might explode. Conversely, if they are less than one, the activation might vanish. Same with the gradients in BP. 

- Partial solution for vanishing/exploding gradient problem: Careful choice of initilization. Concretely, set the input feature for each layer to be mean zero and standard variance. Intuition behind this: Hope all the weights matrix not too much bigger/smaller than 1. 

        For Relu: np.random.randn(shape) * np.sqrt(2/n^{l-1}) . (Setting the variance to be 2/n) 

        For tanh: tanh, the last term becomes np.sqrt(1/n^{l-1}) (Xavier initilization)
        
        Alternative: np.sqrt(2/(n^{l-1}+n^{l}))
 
 - When we do the numeric difference gradient checking, if epsilon is the order of ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/03.gif ""), then if the difference between the numeric and BP is the order of ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/03.gif ""), that's good. If it's the order of ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/04.gif ""), that's bad!!
  
 - Implement Gradient Checking without Dropout!
  
 - Train the network for some time so that w, b can wander away from zero. And then do gradient checking. 
 

 # Optimization Algorithms
 
 
 ## Mini-batch gradient descent
 - Using SGD might cause you losing the speedup from vectorization and will never converge. 
 
 - Choosing Mini-batch size rule:
    1. If small training set(<2000), just use batch GD.
    2. Typical mini batch size: 64, 128, 256, 512.
    3. Make sure mini-batch size fits into your CPU or GPU.
 
 - Exponentially weighted Averages:
    - v0 = 0
    - v1 = 0.9 * v0 + 0.1 * theta1
    - v2 = 0.9 * v1 + 0.1 * theta2
    - ...
    - vt = 0.9 * v(t-1) + 0.1 * theta_t
    
<!---
    $$v_t = \beta v_{t-1} + (1-\beta)\theta_t$$
-->

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/05.gif "")
    
    
 - v(t) is approximately average over <!--- $\frac {1}{1-\beta}$ --> 
![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/06.gif "") days temperature. 
 
 - This way of computing moving average is good both in terms of memory and computational efficiency. 
 
 - It's good to compute the moving average for a range of variables. 
 
 - Bais Correction for Exponentially weighted Average (helps get better estimate early on):

<!---
    $$v_{t}corrected = \frac {\beta v_{t-1} + (1-\beta)\theta_t}{1-\beta^{t}}$$
-->
 
 ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/07.gif "")
 
 ## Gradient Descent with Momentum

Use exponentially weighted average to compute the gradient, and use that gradient to update the parameters.

<!---
$$V_{dW} = \beta V_{dW} + (1-\beta)dW$$
$$V_{db} = \beta V_{db} + (1-\beta)db$$
$$W = W - \alpha V_{dW}$$
$$b = b - \alpha V_{db}$$
-->

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/08.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/09.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/10.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/11.gif "")

## Momentum:
- On iteration t, compute dw, db on current mini-batch. 
<!---
    - $V_{dw} = \beta V_{dw} + (1-\beta) dw$
    - $V_{db} = \beta V_{db} + (1-\beta) db$
    - $w = w - \alpha V_{dw}$
    - $b = b - \alpha V_{db}$
-->
![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/12.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/13.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/14.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/15.gif "")
    
- It smooths out oscillation in the direction that we don't need. Ant at the same time maintaining the gradient that points toward the minimum. 
- The most commonly use <!---$\beta$ --> ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/51.gif "") is 0.9, which means we're averaging out the last ten gradients. 
- When implementing gradient descent with momentum, it's not very often to use bias correction. 
- Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.
    
## RMSprop
<!---
$$S_{dW} = \beta_2 S_{dW} + (1 - \beta_2)dW^2$$

$$S_{db} = \beta_2 S_{db} + (1 - \beta_2)db^2$$

$$W = W - \alpha \frac {dW} {\sqrt S_{dW}}$$

$$b = b - \alpha \frac {db} {\sqrt S_{db}}$$
-->

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/16.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/17.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/18.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/19.gif "")


- On iteration t, compute dw, db on the current mini-batch
    - Sdw = beta2 * Sdw + (1-beta2) * (dw)^2   <-- element-wise square
    - Sdb = beta2 * Sdb + (1-beta2) * (db)^2
    - Keeping a exponentially average of the square of the derivative. 
    - w = w - alpha * dw /(sqrt(Sdw))
    - b = b - alpha * db/(sqrt(Sdb))
- Intuition:
    - If during training, the derivative in the unwanted direction is large(say db is large)
    - Then in the updating equation, we are dividing a relatively large number. 
    - And that helps damp out the oscillation in the unwanted direction. 
    - The derivative in the wanted direction would keep going. 
    
## Adam (Adaptive moment estimation)

Adam puts momentum and RMSprop together. 

1. Initialize:
<!---
$$V_{dW} = 0, S_{dW}=0, V_{dW}=0, S_{db}=0$$
-->
![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/20.gif "")
2. Calculate:
- Momentum

<!--
$$V_{dW} = \beta_1 V_{dW} + (1-\beta_1)dW$$
$$V_{db} = \beta_1 V_{db} + (1-\beta_1)db$$
-->
![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/21.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/22.gif "")


- Momentum with bias correction
<!--
$$V_{dW}corrected = \frac {V_{dW}} {(1-\beta_1^t)}$$

$$V_{db}corrected = \frac {V_{db}} {(1-\beta_1^t)}$$
-->

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/23.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/24.gif "")

- RMSprop

<!---
$$S_{dW} = \beta_2 S_{dW} + (1-\beta_2)dW^2$$
$$S_{db} = \beta_2 S_{db} + (1-\beta_2)db^2$$
-->

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/25.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/26.gif "")

- RMSprop with bias correction

<!--
$$S_{dW}corrected = \frac {S_{dW}} {(1-\beta_2^t)}$$

$$S_{db}corrected = \frac {S_{db}} {(1-\beta_2^t)}$$
-->

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/27.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/28.gif "")

3. Update
<!---
$$W = W - \alpha \frac {V_{dWcorrected}} {\sqrt {S_{dWcorrected} + \epsilon}} $$

$$b = b - \alpha \frac {V_{dbcorrected}} {\sqrt {S_{dbcorrected} + \epsilon}} $$
-->

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/29.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/30.gif "")

- Hyper-parameters:

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/31.gif ""): 0.9 (default for moving average dw )

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/32.gif ""): 0.999 ( ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/33.gif "") )
    
![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/34.gif ""): ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/34b.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/35.gif "") : needs to be tuned. 
    
Some advantages of Adam include:
- Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum)
- Usually works well even with little tuning of hyperparameters (except  αα )


## Learning rate decay

Slowly reduce your learning rate  over time

- As alpha goes smaller, the steps are smaller. Ends up in a tiny region around the minimum. 
- 1 epoch - 1 pass through the data.

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/36.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/37.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/38.gif "")


## Local Optima

- Most of the local optima in deep learning cost function is saddle point. 
- It takes a very very long time to go down to the saddle point before it finds its way down. (Problem of plateaus)
- Use momentum or Adam to solve this kind of problems.


        
# Hyperparameters Tuning

Most important parameter is ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/35.gif "")
Second most important parameters are ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/51.gif ""), number of hidden units and mini-batch size

- Try random search instead of grid search during hyperparameter tuning.

- Coarse to fine scheme. 

- Pick numer of units per layer <!--- $n^{[l]}$ --> ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/39.gif "") and number of layers at random


- If we tried to sample between 0.0001 to 1, do it on the log scale. 
    - r = -4 * np.random.rand()
    - Alpha <!--_ $\alpha = 10^{r}$ --> ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/40.gif "")
    - Generally, if we tried to sample <!--$10^{a}$ - $10^{b}$--> ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/41.gif "")  - ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/42.gif ""), we can calculate <!--$a = log_{10}(leftNum)$--> ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/43.gif ""), <!--$b = log_{10}(rightNum)$--> ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/44.gif "")

5. If we tried to sample between 0.9 - 0.99999 (the beta value for exponentially weighted average)
    - (1-0.9) = 0.1, (1-0.99999)=0.00001
    - sample between r = [-4, -1]

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/45.gif "")


## Batch Normalization

Normalize the activation(z, instead of a) of the previous layer to speed up training for later layers. 
- Given intermediate values z(1),...,z(m) of a specific layer: 

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/46.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/47.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/48.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/49.gif "") (where <!--$\gamma$--> ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/50.gif "") and <!--$\beta$--> ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/51.gif "")  are learnable parameters for each hidden unit. )

- Use <!--$\sim{z} {(i)}$--> ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/52.gif "") instead of z in later computation. 

- Intuition: 
    - It makes weights deeper in the NN more robust to changes in the earlier layers in the NN.
    - It allows each layer of NN to learn a little bit more independently. 
    - It has a slight regularization effect if we use BN on mini-batch, because each mini-batch is a little bit noisy.

- Batch Norm at test time:
    - estimate mean (<!--$\mu$--> ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/53.gif "") ) and variance (<!--$\sigma$--> ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/54.gif "") ) using exponentially weighted average across mini-batch during training. 
    - Use that mean (<!--$\mu$--> ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/53.gif "") ) and variance (<!--$\sigma$--> ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/54.gif "") ) in testing. 

## Softmax Layer

Used for multi-class clasification. Softmax regression generalizes logistic regression to C classes.

- Algorithm:

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/55.gif "")

![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/56.gif "")

- Unlike other activation functions, which takes in a number and output a number. Softmax takes in and output a vector.

- The decision boundary between any of the two classes will be linear. 

- Loss function for Softmax:

    - Loss: ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/57.gif "")
    - Since y can never be greater than 1, then in order to make L small, y should be as close to one as possible. 
    - For the entire training set: <!--$J = \frac {1} {m} \sum_{i=1}^{m} {L}$--> ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/58.gif "")

- Gradient Descent for Softmax:
    - dz = y_hat - y
    - Compute the <!--$\frac {dJ} {dz}$--> ![](https://github.com/SekibOmazic/deeplearning.ai/raw/master/Hyperparameter-tuning_Regularization_Optimization/images/59.gif "")
