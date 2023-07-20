
## Neural network architecture and deep learning


##### Basics of NN (Neuron)
Neural networks is a powerful expressive machine learning architecture for learning arbitrary input-output function given enough training data

Basic building block of a neural network is a neuron:

![[NN architecture]]

Then we can start stacking multiple Neurons, together and then we can stack them together in weird patterns as shown below:
![[diff NN arch]]

And we can build up this complexity such as with this:
***Artificial Neural Network (ANN)***
![[ANN]]
- It is a Network with nodes and edges describing the how the architecture is connected where each node can have their own activation functions
- Stacks the transformation together to provide us with sequential operations 


##### Deep Neural Network (DNN)

Stacking enough ANN in the middle results in 
***Deep Neural Network***
![[DNN]]

There are a bunch of different DNN that people use and their architectures are used for different things:
![[Pasted image 20230720145132.png]]
The different color corresponds to different types of nodes and the different topology speaks to how they are stacked
*Example*: Structures that are bottlenecking are condensing some sort of information into a deeply packed output

* There are more types of architectures and still more being produced by researchers, this is only a small sample

##### Convolutional neural network (CNN)

Convolutional NN are mainly used for image recognition

![[Pasted image 20230720145449.png]]

Basic idea is that it has is that it performs convolution for the output from of the node. 

Each layer might do its own things such as one might be doing edge detection and then passing it off and the other might be looking at the color grouping and then stack the information from these two images

###### Convolutions:
Basically multiplying a function of sort. Visualise a mask that takes one image and slides over it resulting in the  pixel of the output image. 

Like the [3B1B video on convolutions](https://youtu.be/KuXjwB4LzSA) 

##### Recurrent Neural Network (Audio / Temporal)

Basically anything that changes with time and have a loop back to itself

This feedback acts sort of like a memory which helps with data where you have to keep track with time
![[Pasted image 20230720145945.png]]

* The usual ones which go from left to right are ***Feedforward Neural network (FNN)***

##### Autoencoder

It takes high dimensional data and turns into a small condensed data
- Can think of ***PCA as a shallow and linear autoencoder***
![[Pasted image 20230720150145.png]]

Since then, we have also discovered how to add more layers in the middle and have ***Deep Autoencoders***
![[Pasted image 20230720150234.png]]
Here the nodes can have non-linear activation functions which has better congestion and extractions and also has more interpretability
As you can tell, such compressions have been abstracted into $\phi(z)$ and $\psi(z)$  in the pic above


##### Opensource softwares:
```python
import PyTorch
import TensorFlow
import Keras
```



## Deep learning to discover coordinates for dynamics

We are mainly going to be talking about Autoencoder system
![[Pasted image 20230720175800.png]]

so here we have an encoder, $\phi$, and a decoder, $\psi$, that condenses high dimensional data into a very latent space, $z$

In this latent space, $z$, we are looking for a dynamical system $\dot{z}$ which equals $f(z)$ where $f()$ is as simple as possible
![[Pasted image 20230720180202.png]]
such that $\frac{d}{dt}z = f(z)$ 

So it is a 2 part problem
- Finding $\phi$ and $\psi$
- Finding $f(z)$

Example: You have a video of the pendulum swinging so you want to discover that it is governed by $\theta$ of the pendulum and the equation is: $$\ddot{\theta} = - \frac{g}{L}\sin(\theta) - \delta \dot{\theta}$$
![[Pasted image 20230720180738.png]]
- This is the sort of like gopro physics


Challenges:
1. Often equations are unknown or partially known for systems such $\frac{d}{dt}g(x) = f(x)$ and we have to find $f(x)$ given $g(x)$. 
2. Nonlinear dynamics are still poorly understood. Coordinate transformations to linearize dynamics
3. High dimensionality often obscure dynamics where pattern exist that can facilitate reduction

If we have high dimensional data, we can use SVD / PCA / POD to turn it into low dimensional data which relies on pattern present in the data itself
![[Pasted image 20230720181223.png]]

- You can think of such SVD/PCA as a shallow and linear autoencoder
![[Pasted image 20230720181324.png]]
You can see the $U$ and $V$ present in $B = U\Sigma^2V^{T}$ or SVD but in reality, we rely on better autoencoders in real life

We can generalise the autoencoder by making a deep autoencoder:
![[Pasted image 20230720150234.png]]
Each node will have a non linear activation function and what this allows us to do is that we can learn non-linear manifold instead of a linear subspace
* This massively decreases DOF (degrees of freedom)

We try to reduce the error between $\phi(x):X\to z$ and $\phi(z): z\to X$ as much as possible.

We are also going to be training these autoencoder networks to learn the dynamical system where it evolves

![[Pasted image 20230720185546.png]]

Changing the coordinates can make a lot of the system simpler to understand such as looking at the planet's orbits through Earth's perspective, ***Geocentric system***, and through Sun's perspective, ***Heliocentric System***. 

![[Pasted image 20230720185748.png]]
Learning the Heliocentric System is a lot easier to understand

* Getting the right coordinate system is essential to understanding the dynamics correctly

##### Koopman review

So instead of condensing down the information, we can increase it to a higher dimension to find a new coordinate system that may suite it best
![[Pasted image 20230720190717.png]]
So you expand through non-linearization to space where linearization is valid
![[Pasted image 20230720190839.png]]

