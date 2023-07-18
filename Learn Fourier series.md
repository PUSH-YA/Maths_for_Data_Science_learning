
## Fourier series part 1 & 2

#### Part 1
Approximate a function with a bunch of sine / cosine waves

Ex:
![[fourier1]]

$$f(x) = \frac{A_{0}}{2} + \sum_{k=1}^{\infty} A_{k}\cos(kx)+B_{k}\sin(kx)$$

The first term is a constant and $A_{k}$ and $B_{k}$ are fourier coefficients

$$A_{K} = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos(kx) \, dx = \frac{1}{\parallel\cos(kx)\parallel^2}<f(x),\cos(kx)>$$
$$B_{k} = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x)\sin(kx) \, dx = \frac{1}{\parallel\sin(kx)\parallel^2}<f(x),\sin(kx)> $$
Coefficients are determined by the Hilbert space inner product of the function $f(x)$ and the corresponding sinusoidal wave. 

##### Hilbert space inner product
It is just multiplication of 2 functions as if they are vectors (computationally), a lot more theory behind it but for my understanding:
![[Pasted image 20230717112611.png]]

##### ***The vector notation:***
The coefficients are just normalised projections of $f(x)$ onto the $\cos(kx)$ or $\sin(kx)$

We can use the projections to denote a function as made up of those projection and orthogonal vectors so,
$$\vec{f} = <\vec{f},\vec{u}>\frac{\vec{u}}{\parallel u \parallel^{2}} + <\vec{f},\vec{v}>\frac{\vec{v}}{\parallel v \parallel^{2}}$$
$$\vec{f} = \frac{1}{\parallel\vec{\cos(kx)}\parallel^{2}}<f(x),\vec{\cos(kx)}>+\frac{1}{\parallel\vec{\sin(kx)}\parallel^{2}}<f(x),\vec{\sin(kx)}> $$

  * ***NOTE: Fourier series is just writing a function in the orthogonal basis of cosine and sine waves***
The basis are indeed orthogonal for different k's but can't find the video where he proved it

* ***NOTE: Really good for approximation such as the Taylor series by having*** $n < \infty$



#### Part 2

if the function's range changes from $(-\pi, \pi)$ to $(0,l)$, then it changes as following

$$f(x) = \frac{A_{0}}{2} + \sum_{k=1}^{\infty} A_{k}\cos\left( \frac{2\pi kx}{l} \right)+B_{k}\sin\left( \frac{2\pi kx}{l} \right)$$

$$A_{K} = \frac{2}{l} \int_{0}^{l} f(x) \cos\left( \frac{2\pi kx}{l} \right) \, dx$$
$$B_{K} = \frac{2}{l} \int_{0}^{l} f(x) \sin\left( \frac{2\pi kx}{l} \right) \, dx$$

the approximation of the function is also periodic for the range $(0,l)$ as shown in the image

![[fourier 2]]



## Discrete Fourier Transform

* This should be called the Discrete Fourier Series because we are looking at the summation rather than the integral which will lead to FFT
 
DFT is a matrix multiplication whereas FFT is a computationally efficient way computing the DFT

Usually you don't have a function at the point but rather a bunch of discrete data at several different points on the x-axis
![[DFT]]

##### What is a Discrete Fourier Transform

DFT will obtain a new data vector from the data vector mentioned above
$$\begin{bmatrix}
f_{1} \\
f_{2} \\
\vdots \\
f_{n}
\end{bmatrix} \to \begin{bmatrix}
\hat{f_{1}} \\
\hat{f_{2}} \\
\vdots \\
\hat{f_{n}}
\end{bmatrix}$$
The transformation is as following
$$\hat{f_{k}} = \sum_{j=0}^{n-1} f_{j} \, e^{-i 2\pi\frac{j}{ n}}$$
where $k$, $j$ and $n$ are $k$'th term, $j$'th iteration and total number of points respectively. The inverse transformation is as following
$$f_{k} = \sum_{j=0}^{n-1} \hat{f_{j}} \, e^{i 2\pi\frac{j}{ n}}$$
where  $\omega_{n} = e^{-2\pi \frac{i}{n}}$ which corresponds to the fundamental frequency of the sines and cosines that will make up the transformation (used in both transformation) and it is  multiplied to each value
$$[f_{1},f_{2}\dots f_{n}] \xRightarrow{DFT} [\hat{f_{1}}, \hat{f_{2}} \dots \hat{f_{n}}]$$


##### Computation

Now we don't want to compute the $f_{k}$ for each value otherwise that will be 2 big for loops ,$O(n^2)$.

so, we will what the values should be for different $k$'s such that the summation can be described with a simple matrix multiplication
$$\hat{f_{k}} = \sum_{j=0}^{n-1} f_{j} \, e^{-i 2\pi\frac{j}{ n}}$$ $$\begin{bmatrix}
\hat{f_{1}} \\
\hat{f_{2}} \\
\hat{f_{3}} \\
\vdots \\
\hat{f_{n}}
\end{bmatrix} = 
\begin{bmatrix}
1 & 1 & 1 & \dots & 1 \\
1 & \omega_{n} & \omega_{n}^2 & \dots & \omega_{n}^{n-1} \\ \\
 1 & \omega_{n}^{2} & \omega_{n}^2 & \dots & \omega_{n}^{2(n-1)}\\ 
 &  &  \vdots &  &  \\
1 & \omega_{n}^{n-1}  & \omega_{n}^{2(n-1)} & \dots & \omega_{n}^{(n-1)^{2}}
\end{bmatrix}\begin{bmatrix}
f_{1} \\
f_{2} \\
f_{3} \\
\vdots \\
f_{n}
\end{bmatrix}$$
* You can notice the pattern here of how the $\omega_{n}$ is changing with each row for each k value

It is a complex matrix and returns a complex vectors where magnitude is amplitude and the angle tells u the phase $z \in \mathbb{C}, z = a+bi$
- magnitude is $\sqrt{ a^{2} +b^{2}}$ and angle is $\angle\tan\left( \frac{a}{b} \right)$

## FFT (Fast Fourier Transform) algorithm

Uses of FFT:
- Derivatives $\implies$ Solve PDE's
- Denoise data
- Analysis of data
- Compression of Audio and images

## Fast Fourier transform (FFT)

## Fourier analysis



