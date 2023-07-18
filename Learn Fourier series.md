
## Fourier analysis

It is the chapter 2 of the [data driven science and engineering](https://faculty.washington.edu/sbrunton/databookRL.pdf)

It is one of the most important and ubiquitous transformation in mathematics, physics and engineering. 

Most of our breakthroughs started with a coordinate transformation and Fourier Transformation is a also a coordinate transformation for representing data / images and so on.


Fourier derived this transformation as a way of approximating the solution to PDE's
he was interested in the heat equation $u(x,y,t)$ where $x,y,t$ are position in $x-axis$, $y-axis$ and $time$.  Laplacian operator:
$$u_{t} = \alpha \nabla^{2}u$$
* He discovered that the Laplacian operator had eigen values, $\lambda$, and eigen vectors, $\vec{v}$, similar to other linear operators where the eigen vectors are sines and cosines with the fundamental frequency , $\omega_{n}$, determined by the boundary conditions and so on.

Since then it has been used for image compression and other data transformations. In fact $SVD$ can be through of something like Data Driven $FFT$ 

##### Connection between FFT and SVD

1. The FFT transforms a signal from the time domain to the frequency domain, revealing the frequencies present in the signal. Similarly, SVD transforms a data matrix from the data space to a new representation space spanned by the singular vectors. This transformation can unveil underlying patterns and structures in the data.
2. The FFT provides the amplitude and phase components of different frequencies in a signal. Similarly, the SVD provides the singular values (related to the amplitudes) and the left and right singular vectors (analogous to the phase) associated with the data.
3. The amplitudes obtained from the FFT can indicate the importance or strength of different frequencies in the signal. Similarly, the singular values obtained from the SVD can represent the importance or relevance of the corresponding singular vectors in the data.


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

##### Idea of how it works

So, the time complexity  of FFT is $\theta(n\log(n))$ whereas DFT is $\theta(n^{2})$  

So let's say you have $n = 2^{10}= 1024$ then the computation would be something like $\hat{f} = F_{1024}f$  however we can make this computation lighter as following
$$\hat{f} = F_{1024}f = \begin{bmatrix}
I_{512} - D_{512} \\
I_{512} - D_{512}
\end{bmatrix}\begin{bmatrix}
F_{512} & 0 \\
0 & F_{512}
\end{bmatrix}\begin{bmatrix}
f_{even} \\
f_{odd}
\end{bmatrix}$$
where $$D_{512} = \begin{bmatrix}
1 & 0 & 0 & \dots  & 0 \\
0 & \omega_{n} & 0 & \dots & 0 \\
0 & 0 & \omega_{n}^{2} & \dots & 0 \\
0 & 0 & 0 & \dots & \omega_{n}^{511}
\end{bmatrix}$$
* This works because you are reshuffling the half of the rows above and below which is what the above computation is doing to the matrix

This is a lot more efficient because instead of $1024$ matrix, now you have to deal with two $512$ matrix which is more efficient cuz matrix mult is $O(n^2)$

We can split the even and odd further on such as the following
$$\begin{matrix} \\
0 & &  0 &  & 0 \\
1 & &  2 &  & 4\\
2   &  &4 &  & 8\\
3  &  & 6 &  & 2\\
4  &  \to & 8 & \to & 6\\ 
5  &  & 1 &  & 1\\
6  &  & 3 &  & 5\\
7  &  & 5 &  & 3\\
8  &  & 7 &  & 7\\
\end{matrix}$$

and basically $F_{1024} \to F_{512} \to F_{256} \to \dots \to F_{4} \to F_{2}$
Now, you are only left with a $2 \times 2$ matrix which is just 4 numbers and we are basically taking ***advantage of the symmetry in the matrix***

* Symmetric matrices are when $A^{T}= A$






## Denoising data w/ FFT (Python)

First lets set up the environment

```python
import numpy as np
import matplotlib.pyplot as plt

#Create a simple signal with two frequencies
dt = 0.001
t = np.arange(0,1,dt)
f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t) # sum of 2 freq

f_clean = f
f = f+2.5*np.random.randn(len(t)) # Add some noise

#plot
plt.figure(figsize=(13,6))
plt.plot(t,f,color = 'c', linewidth = 1.5, label = 'Noisy')
plt.plot(t,f_clean,color = 'k', linewidth = 2, label = 'Clean')
plt.xlim(t[0], t[-1])
plt.legend()
```

![[Pasted image 20230718210624.png]]

First we will compute the FFT, which is straight forward in almost all programming languages
```python
## compute the FFT
n = len(t)
fhat = np.fft.fft(f,n) # compute the fft

## power spectral density (power bc norm*conj per frequency)
PSD = fhat * np.conj(fhat) / n # power spectrum

freq = (1/(dt*n)) *np.arange(n) # create x-axis for freq
L = np.arange(1,np.floor(n/2), dtype = 'int') # only plot the first half

fig,axis = plt.subplots(2,1, figsize = (14,7))

plt.sca(axis[0])
plt.plot(t,f,color = 'orange', linewidth = 2, label = "Noisy")
plt.plot(t,f_clean,color = 'blue', linewidth = 2, label = "Clean")
plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axis[1])
plt.plot(freq[L], PSD[L], color = 'orange', linewidth = 2, label = "Noisy")
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()

plt.show()
```

![[Pasted image 20230718212328.png]]

Based on this, we can tell that the power plot has 2 clean peaks: 50hz && 120hz

We can use this to defilter the data and then inverse Fourier transform

$$f \xrightarrow{FFT} \hat{f} \xrightarrow{filter} \hat{f}_{\text{filt}} \xrightarrow{iFFT} f_{\text{filt}}$$

```Python
### plot EVERYTHING!!!!

fig,axis = plt.subplots(3,1, figsize = (14,7))

plt.sca(axis[0])
plt.plot(t,f,color = 'orange', linewidth = 2, label = "Noisy")
plt.plot(t,f_clean,color = 'blue', linewidth = 2, label = "Clean")
plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axis[1])
plt.plot(freq[L], PSD[L], color = 'orange', linewidth = 2, label = "Noisy")
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()

plt.sca(axis[2])
plt.plot(freq[L], PSD[L], color = 'orange', linewidth = 2, label = "Noisy")
plt.plot(freq[L], PSDClean[L], color = 'blue', linewidth = 1.5, label = "filtered")
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()

plt.show()
```

![[Pasted image 20230718213601.png]]