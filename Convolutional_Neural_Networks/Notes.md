# Edge detection

1. Vertical edge detection

- Use 3 x 3 matrix (filter or kernel):

    1 0 -1

    1 0 -1
    
    1 0 -1
    
    - bright pixel on the left, dark one on the right

- Multiply (convolve) input matrix (image) with the filter to obtain output matrix (image)
  - Multiplication is sum of an elementwise product 

2. Horizontal edge detection

- Use 3 x 3 matrix (filter or kernel):
    
     1  1  1

     0  0  0
    
    -1 -1 -1

3. Other filters

    1 0 -1

    2 0 -2   - Sobel filter

    1 0 -1

    3  0 -3

    10 0 -10  - Schorr filter 

    3  0 -3

Using backpropagation a NN can learn all of these 9 parameters (w1, w2, ..., w9) to create new filters

# Padding

- Problem:

    - Image (n,n), filter (f,f) then output image has dimensions (n-f+1, n-f+1) - output shrinks.
    - Pixels in the corners or edges are used less in the output.

- Solution:
    - pad the image with the additional border of 1 pixel all around the edges (p = padding)

- Valid (no padding) convolution: (n,n), (f,f) -> (n-f+1, n-f+1)
- Same convolution (output size same as input size): p = (f -1) / 2

Size of filter is usually odd number.

# Strided convolution

Apply filter not on the next pixel but jump over one step.

- Image (n, n), filter (f, f) padding p and stride s = 2 leads to the output size:
    - ( (n +2p -f)/s + 1, (n +2p -f)/s + 1 )
- Filter should be entirely inside image (incl. padding)

- Real convolution would involve flipping the filter horizontaly and vertically. Without flipping this operation is cross-correlation. But in ML we just don't use flipping of filters.


# Convolutions over volume

### Convolution over RGB images

Images have height, width and number of channels, so the filter also have same structure.

Number of channels in the image must match number of channels in the filter.

Example:

(6 x 6 x 3) * (3 x 3 x 3) = ( 4 x 4 )

### Applying multiple filter

- apply first filter:
    - (6 x 6 x 3) * (3 x 3 x 3) = ( 4 x 4 )
- apply scond filter:
    - (6 x 6 x 3) * (3 x 3 x 3) = ( 4 x 4 )
- stack the outputs:
    - (4 x 4 x 2)

Summary:

(n x n x nc) * (f x f x nc) = ( n - f + 1 x n -f + 1 x nc') 

nc = number of channels
nc' = number of applied filters

### One layer of a Convolutional network

- Example: 10 filters, all 3 x 3 x 3 in one layer. How many parameters does that layer have?
    - each filter has 3 x 3 x 3 plus bias = 28 parameters, so in total 280 parameters

$f^{[l]}$: filter size
$p^{[l]}$: padding
$s^{[l]}$: stride

Input: $n^{[l-1]}_{H}$ x $n^{[l-1]}_{W}$ x $n^{[l-1]}_{c}$

Output: $n^{[l]}_{H}$ x $n^{[l]}_{W}$ x $n^{[l]}_{c}$

$$ n^{[l]} = \frac {n^{[l-1]} + 2p - f^{[l]}} {s^{[l]}} + 1$$

$n^{[l]_c}$: number of filters

Each filter is: $f^{[l]}$ x $f^{[l]}$ x $n^{[l-1]}_{c}$

Activations: $a^{[l]}$ = $n^{[l]}_{H}$ x $n^{[l]}_{W}$ x $n^{[l]}_{c}$

Using Batch Gradient Descent activation is: $A^{[l]}$ = m x $n^{[l]}_{H}$ x $n^{[l]}_{W}$ x $n^{[l]}_{c}$

Weights: $f^{[l]}$ x $f^{[l]}$ x $n^{[l-1]}_{c}$ x $n^{[l]}_{c}$

Bias: $n^{[l]}_c$


# Simple Convolutional Network example

Example image: 

$n^{[0]}_H$ = $n^{[0]}_W$ = 39, 

$n^{[0]}_c$ = 3, 

$f^{[1]}$ = 3, 

$s^{[1]}$ = 1, 

$p^{[1]}$ = 0

10 filters

Output will be: $a^{[1]}$ = 37 x 37 x 10

Then for the next layer use e.g.

$f^{[2]}$ = 5, 

$s^{[2]}$ = 2, 

$p^{[2]}$ = 0

20 filters

Output will be: $a^{[2]}$ = 17 x 17 x 20

## Types of layer in a convolutional network
- Convolution (Conv)
- Pooling (Pool)
- Fully Connected (FC)


# Pooling layers

### Max Pooling

- Divide your image into regions and take the max value of each region as the output

- Example:
    - input image 4 x 4, f = 2, s = 2 Output will be 2 x 2

- Intuition:
    - Check if there is a feature detected in a region
- Max pooling has no parameters to learn

### Average pooling

- Instead of using maxes just use averages.

### Summary
- Hyperparameters:
    - f: filter size (usually 2)
    - s: stride (usually 2)
    - Max or Average pooling

# CNN Example

32x32x3, f=5, s=1 

|

28x28x6 (conv1) 

| maxpool, f=2, s=2

14x14x6 (Pool1)

| f = 5, s = 1

10x10x16 (Conv2)

| maxpool, f=2, s=2

5x5x16 (Pool2)

| flatten into one vector of 400 units (5x5x16)

| $W^{[3]}$ (120, 400), $B^{[3]}$ (120)

FC3 (120 units)

| $W^{[4]}$ (84, 120), $B^{[4]}$ (84)

FC4 (84 units)

|

Softmax (10 units)

# Why convolutions?

- Advantages: 
  - Parameter sharing
  - sparsity of connections

So you can train the network with less data and it is less prone to overfitting.

Cost 𝐽 = $\frac {1} {m} \sum_{i=1}^{m} L(Yhat^{i}, Y^{i})$

Use gradient descent to optimize parameters to reduce 𝐽 

# Case studies

- Classic networks
  - LeNet-5
  - AlexNet
  - VGG

- ResNet

- Inception

## Classic Networks

1. LeNet-5

(32,32,1)

| 6 filters (5,5), s=1

(28,28,6)

| avg pool, f = 2, s = 2

(14, 14, 6)

| 16 filters (5,5), s=1

(10,10,16)

| avg pool, f = 2, s = 2

(5,5,16)  --> Flattened to 400 units vector

| fully connect to

(1,120) 

| fully connect to

(1,84)

| (softmax)

Yhat

This network has ca. 60K parameters

$n_W$ and $n_H$ shrink, $n_C$ grows

- Typical architecture:
  - conv -> pool -> conv -> pool -> fc ->fc -> output

- Advanced notes:
  - earlier sigmoid/tanh
  - recently ReLU

2. AlexNet

(227,227,3)

| 96 filters (11,11), s = 4

(55,55,96)

| Max pool (f=3, s = 2)

(27,27,96)

| (5,5) same convolution

(27,27,256)

| Max pool (f=3, s = 2)

(13,13,256)

| (3,3) same convolution

(13,13,384)

| (3,3) same convolution

(13,13,384)

| (3,3) same convolution

(13,13,256)

| Max pool (f=3, s = 2)

(6,6,256) --> flttens to 9216 units vector

| fully connect

(4096)

| fully connect

(4096)

| softmax 

Yhat (1000 units)

Similar to LeNet but much bigger (60M parameters)

Uses ReLU for activation, multiple GPUs


3. VGG - 16

- CONV = 3x3 filter, s=1, same
- MAX POOL = 2x2, s=2

(224, 224,3)

| [CONV 64] (2 layers)

(224,224,64)

| POOL 

(112,112,)

| [CONV 128] (2 layers)

(112,112,128)

| POOL 

(56,56,128)

| [CONV 256] (3 layers)

(56,56,256)

| POOL 

(28,28,256)

| [CONV 512] (3 layers)

(28,28,512)

| POOL

(14,14,512)

| [CONV 512] (3 layers)

(14,14,512)

| POOL

(7,7,512)

| fully connect

(4096)

| fully connect

(4096)

| softmax 

Yhat (1000 units)

Large network with ca. 138M parameters

## Residual Network (ResNet)

Built out of residual blocks

- main path (for 2 layers network)

  - $a^{[l]}$ --> linear --> ReLU ($a^{[l+1]})$ --> linear --> ReLU ($a^{[l+2]}$)

  - $z^{[l]} = W^{[l]}a^{[l]}+b^{[l]}$
  - $a^{[l+1]} = g(z^{[l+1]})$
  - $z^{[l+2]} = W^{[l+2]}a^{[l+1]}+b^{[l+2]}$
  - $a^{[l+2]} = g(z^{[l+2]})$

- shortcut 
  - $a^{[l]}$ --> ReLU ($a^{[l+2]}$)
  - $a^{[l+2]} = g(z^{[l+2]} + a^{[l]})$

- With plain networks as you increase number of layers the training error tends to decrease after while but then thends to go back up

- with Resnet the training error keeps going down

## Why ResNet work?

Learning identity function is easy for residual block to learn!

Usually uses same convolution

## Networks in networks and 1x1 convolution

Can use it to reduce numbre of channels.

- Example:
  - (28,28,192) --> CONV 1x1x32 --> (28,28,32)


## Inception network motivation

When designing a layer for ConvNet, you have to decide 1x3, 3x3 or 5x5 filter, or do you need pooling. Inception network uses them all!

- Example:
  - (28, 28, 192)
  - use (1x1x64) --> (28,28,64)
  - also use (3x3x128), same convolution --> (28,28,128)
  - also use (5x5x32), same convolution --> (28,28,32)
  - also use MaxPool (28, 28,32), s = 1 --> (28,28,32)
  - stack all results together

The problem of incetion network is big computational cost

This can be reduced by using 1x1 convolution to reduce number of channels and then run your filter (e.g. 5x5) to much smaller volume (also called "bottleneck layer")


## Inception network

Bunch of inception modules repeated.


# Practical advices for using ConvNets

- Use open source implementations

- Transfer learning:
  - use sommeone elses implementation and train only last (softmax) layer. So freeze earlier layers.
  - precompute input on all frozen layers and save it to disc. Then use saved results to train softmax layer.
  - If you have a lot of training data you could freeze not all but just a few layers and train the network with later layers and you own sofmax layer.

- Data augmentation
  - used to improve performance
  - common augmentation method is mirroring
  - another method is random cropping
  - also use rotation or sheering
  - color shifting (add/subtract values on RGB channels)

- State of computer vision
  - two sources of knowledge: labeled data and hand engineered features
  - use open source implementation if possible
  - use pretrained models and fine-tune on your dataset

# Detection Algorithms

### Object localization

- Image classification: what is in the image
- Classification with localization: what object is in the image and where in the image is this object (usually one)
- Detection: multiple objects

- Classification with localization: have the net outputs classes (with softmax) and the bounding box of the object (bx, by, bh, bw)
  - bx, by - center of the bounding box
  - bw, bh - size of the bounding box

- Target label:
  - [Pc, bx, by, bw, bh, c1, c2, c3, ...]
  - Pc - probability that there is an object
  - c1 - class 1
  - c2 - class 2

- Loss:

$$ L(Yhat, Y) = 
\begin{cases}
(Yhat_{1}-Y_{1})^2 + ... + (Yhat_{n}-Y_{n})^2,& \text{if Y1 = 1 }\\
(Yhat_{1}-Y_{1})^2 & \text{if Y1 = 0 }
\end{cases}
$$

### Landmark detection

- Output vector:
  - [P, l1x, l2y, ... lnx, lny]
  - l - landmark coordinates
- Landmarks must be consistent across images

### Object detection

- Sliding window detection
  - choose the size of the sliding window
  - slide windown across the input image
  - run ConvNet for each position of the sliding window to classify 
  - resize the sliding window (make bigger) and repeat
  - Disadvantage: huge computation cost and with bigger stride you hurt the performance

### Convolutional Implementation of Sliding Windows

- Turning FC layers into convolutional layers
- example:
  - (14,14,3) ---> 16 f=5 --> (10,10,16) ---> max pool 2x2 ---> (5,5,16) ---> FC (400) ---> FC (400) --> softmax (4)
  - Instead of using fully connected layer (FC400) apply 400, f=5 (400 5x5 filters) to get (1,1,400) volume
  - for the next FC layer use 400 1x1 filters to get (1,1,400) volume
  - output is now (1,1,4) volume!

- Convolutional implementation allows you to share computations

### Bounding box prediction

- output accurate bounding box with YOLO algorithm
- YOLO:
  - put a grid over the image
  - apply object classification for each grid cell
  - YOLO takes midpoint of the object and assings it to the grid cell containing this point
  - Target output is a volume of e.g. (3,3,8) - for the 3x3 grid and output vector of 8 elements

- Bounding box sizes are specified relatively to grid cell

### Intersection over Union

- compute intersection over union (IoU) of predicted and actual bounding box
- "Correct" if IoU >= 0.5
- IoU is a measure of the overlap of two bounding boxes

### Non-max Suppression

- With many grid cells you can end up with multiple detections per object.
- Take the one with the biggest probability and suppress all the ones with high overlap (IoU) with the chosen one.

Algorithm:

1. Discard all boxes with Pc <= 0.6
2. While there are any remaining boxes:
  - Pick the box with the largest Pc and output that as a prediction
  - Discard any remaining box with IoU >= 0.5 with the output box from the previous step

### Anchor Boxes

- For multiple object use anchor boxes

- Output vector is [Pc, bx, by, bh, bw, c1, c2, c3, Pc, bx, by, bh, bw, c1, c2, c3]

- With 2 anchor boxes:
  - Each object in training image is assigned to grid cell that contains object's midpoint and anchor box for the grid cell with highest IoU

# Face recognition

- Verification (1:1) 
  - Input image, name/ID
  - Output whether the input image is that of the claimed person
- Recognition (1:K)
  - has a database of K persons
  - get an output image
  - output ID if the image is any of the K persons (or "not recognized")

### One Shot Learning

Learning from one example to recognize the person again

- Learing a "similarity" function
  - d(img1, img2) = degree of difference between images
  - if d(img1, img2) > t then "same" else "different"


### Siamese Network

- feed the network with one image x1 and get let's say 128 unit vector f(x1) as an encoding of x1
- for the second image do the same - f(x2)
- $d(x1, x2) = || f(x1) - f(x2) ||^{2}_{2}$
- parameters of NN define an encoding $f(x^{(i)})$
- learn parameters so that:
  - if $x^{(i)}$, $x^{(j)}$ are the same person, $|| f(x^{(i)}) - f(x^{(j)}) ||^{2}$ is small
  - if $x^{(i)}$, $x^{(j)}$ are the different persons, $|| f(x^{(i)}) - f(x^{(j)}) ||^{2}$ is large

### Triplet Loss

Distance between Anchor (A) image and Positive (P) image should be small. Distance between Anchor (A) and Negative (N) should be large.

$$ || f(A) - f(P) ||^{2} <= || f(A) - f(N)||^{2}$$

To be sure that NN doesn't output 0s we change this equation to:

$$ || f(A) - f(P) ||^{2} - || f(A) - f(N)||^{2} + \alpha <= 0$$

$\alpha$ is a margin

Loss function (given 3 images A, P, N):

$$L(A,P,N) = max(|| f(A) - f(P) ||^{2} - || f(A) - f(N)||^{2} + \alpha, 0)$$

Cost function:

$$ J = \sum_{i=1}^{m} {L(A^{(i)}, P^{(i)}, N^{(i)})}$$

For training you need multiple set of pictures of the same person.

- During training if A, P, N are chosen randomly, d(A,P) + $\alpha$ <= d(A,N) is easily satisfied

- choose triplets that are hard to train on
  - d(A,P) $\approx$ d(A,N)

### Face Verification and Binary Classification

For binary classification use element wise difference in absolute values between 2 encodings:

$$ Yhat = \sigma {(\sum_{k=1}^m {| w_i f(x^{(i)})_k - f(x^{(j)})_k + b_i|})} $$


# Neural Style Transfer

### What are deep ConvNets learning?

Pick a unit in layer 1. Find 9 image patches that maximize the unit's activation.
Repeat for other units.

### Cost Function

$$ J(G) = \alpha J_{content}(C, G) + \beta J_{style}(S, G)$$

1. Initiate G randomly
  - G: 100 x 1000 x 3
2. Use gradient descent to minimize J(G)
  - $G = G - \frac {\delta} {\delta G} J(G)$

### Content Cost Function

- say you ue hidden layer l to compute content cost
- use pre-trained ConvNet (e.g. VGG)
- Let $a^{(l)[C]}$ and $a^{(l)[G]}$ be the activation of layer l on the images
- If $a^{(l)[C]}$ and $a^{(l)[G]}$ are similar both images have same content

$$ J_{content} = \frac {1}{2}|| a^{(l)[C]} - a^{(l)[G]}||^2$$


### Style Cost Function

- Say you are using layer l's acivation to measure "style".
- Define style as correlation between activation across channels.
- Example: 1 channel detects vertical lines and second channel detects orange colors. They are correlated if one part of image has vertical lines that part will probalby have orange tint.
- Style matrix
  - let $a^{[l]}_{i,j,k}$ = activation at (i,j,k). $G^{[l]}$ is $n_c^{[l]}$ x $n_c^{[l]}$

$$G^{[l](S)}_{kk'} = \sum^{n_H^{[l]}}_{i=1} \sum^{n_W^{[l]}}_{j=1} a^{[l](S)}_{ijk} a^{[l](S)}_{ijk'}$$

$$ k' = 1, ..., n_C$$

$$G^{[l](G)}_{kk'} = \sum^{n_H^{[l]}}_{i=1} \sum^{n_W^{[l]}}_{j=1} a^{[l](G)}_{ijk} a^{[l](G)}_{ijk'}$$

$$J_{style}^{[l]}(S, G) = \frac {1}{(2n^{[l]}_H n^{[l]}_W n^{[l]}_C)^2} || G^{[l](S)} - G^{[l](G)}||_{F}^{2}$$

$$J_{style}(S, G) = \sum_{l} {\lambda^{[l]} J_{style}^{[l]} (S,G)} $$

$$J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$$

### 1D and 3D Generalizations

