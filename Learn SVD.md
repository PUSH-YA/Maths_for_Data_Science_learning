## Overview

Stands for ***Singular Value Decomposition*** and is used for ***_data or dimension reduction_*** 
Ex: High resolution images

* ***Note: think of it more like data driven generalisation of Fourier transform***
* It allows us to ***tailor*** a coordinate system based on the information we have 

##### Use cases:
1. Solve $Ax = b$ for non square A -> least square linear regression 
2. Basis for ***Principal Component analysis*** or ***PCA*** correlation
3. The Google's page rank algorithm uses it
4. Also used in recommender system in Facebook and Netflix

Basically: most useful cases for Linear algebra *(Matrix decomposition)* & also very **simple** and **interpretable**

##### Extracting patterns:
1. We can use it to preserve the most dominant factors in data while disregarding the less substantial ones
2. By keeping only the most significant singular values and their associated singular vectors, you can represent the data using fewer values
3. the singular values associated with the noise components tend to be smaller compared to the singular values representing the signal. By setting the smaller singular values to zero or removing them, you can suppress the noise and reconstruct a cleaner version of the data.
4. ***MOST IMPORTANT:*** The singular vectors obtained from SVD can provide insights into the underlying patterns in the data. The columns of the U matrix represent the left singular vectors, which can be interpreted as basis vectors that capture the most significant patterns or modes of variation in the data. These patterns can reveal relationships, clusters, or important features that are inherent in the dataset.
5. SVD allows you to reconstruct the original data matrix using a subset of singular values and vectors. By retaining only the dominant singular values and their corresponding singular vectors, you can reconstruct an approximation of the original data.

## Mathematics

Let $$X = \begin{bmatrix}
X_{1} & X_{2} & X_{3}  & X_{m}\\
\vdots & \vdots & \vdots  & \vdots \\
\end{bmatrix} \text{where } X_{i} \in \mathbb{R}^n $$ 
representing high dim data where $X_{i}$ represents information for one example such as 1 person... that is ***evolving in time*** so, $X_{1} \to X_{2} \to X_{3} \dots$

SVD is basically $X = U\Sigma V^T$  where 

* $U$ is $n \times n$ matrix $$U = \begin{bmatrix}
u_{1} & u_{2} & u_{n} \\
\vdots & \vdots & \vdots \\
\end{bmatrix} \text{where } U_{i} \in R^{n}$$
	It is an orthogonal matrix so, $U U^{T}= U^{T}U=I_{n \times n}$
	*Side note: orthogonal matrix is where any 2 dist row / col dot product is 0.*
	
	$U_{i}$ are eigenfaces of $X$ that hierarchically arranged with $U_{1} > U_{2}$ 
	Each $U_{i}$ corresponds $X_{i}$
	
	* ***NOTE: U contains information about column space***

* $\Sigma$ is a diagonal matrix with $\sigma \in \mathbb{R}^n$ the rest rows are emtpy. 
	There are m non zero singular values $$\Sigma = \begin{bmatrix}
	\sigma_{1} & \dots & \dots & \dots \\
	 \dots & \sigma_{2} & \dots & \dots \\
	 \dots & \dots & \sigma_{3} & \dots \\
	\dots & \dots & \dots & \sigma_{m} \\ \\
	0 & 0 & 0 &  0 \\
	0 & 0 & 0 & 0
	\end{bmatrix}$$
	Here the entries are also hierarchically arranged so $\sigma_{1} > \sigma_{2} \geq 0$ 

* $V$ is $m \times m$ matrix transposed $$V = \begin{bmatrix}
V_{1} & V_{2} & V_{3}  &  V_{m}\\
\vdots & \vdots & \vdots & \vdots
\end{bmatrix}$$
	$V$ is orthogonal matrix so: $VV^{T}= V^{TV}= I_{m \times m}$
	
	<i><b><span style="color:#ffb6c1">NOTE: v contains information about row space</span></b></i>


	This one is a bit tricky cause it is transposed so better to visualise it as $$V^T = \begin{bmatrix}
	V_{1,1} \dots V_{1}^T \dots\\
	V_{2,1} \dots V_{2}^T \dots \\
	\dots \,\dots \dots\dots\\
	V_{m,1} \dots V_{m} \dots \\
	\end{bmatrix}$$
	So each entry in $V_{k,i}$ corresponds to $X_{i}$ so where $k \in \{1,m\}$ such as the ***first column above***
	
	<i><b><span style="color:lightpink">Side note: The hierarchical also corresponds to the original X matrix being arranged with time flow</span></b></i>

*These matrix decomposition exists in `R`, `Python`, `Julia`  and other sci-comp languages*
```MATLAB
[U,S,V] - svd(X)
```

<b><i><span style="color:lightpink"> Note: each matrix decomp is unique</span></i></b>

## Dominant correlations

you can think of $U$ and $V$ as eigenvectors of the correlation matrix $XX^{T}$ or $X^{T}X$

If $X$ is tall skinny matrix, then the correlation matrix would look like $$X^{T}X= \begin{bmatrix}
\dots X_{1}^{T}\dots \\
\dots X_{2}^{T} \dots \\
\end{bmatrix}_{m \times n} \begin{bmatrix}
\vdots & \vdots \\
X_{1} & X_{2} \\
\vdots & \vdots
\end{bmatrix}_{n \times m} = \begin{bmatrix}
X_{1}^{T}X_{1} & X_{1}^{T}X_{2}  & \dots \\
X_{2}^{T}X_{1} & X_{2}^TX_{2} & \dots \\
\vdots & \vdots & \vdots
\end{bmatrix}$$
Each product in the above matrix multiplication is $X_{i}^{T}X_{j} = <X_{i},X_{j}>$
	Correlation matrix -> So, Small $m \times m$ matrix contains inner product and all that information ***(dense information)***, essentially guarantees to have non-negative real eigen values

$X^{T}X= (V\Sigma U^{T}) (U\Sigma V^{T})= V\Sigma (UU^{T})\Sigma V = V \Sigma^{2} V^T$
	So this is an eigen value decomposition where $\Sigma^2$ are eigen values and **$V$ are the eigen vectors***
***

Similarily: U has eigen vectors for $XX^T$
	$XX^{T}= (U\Sigma V^{T}) (V\Sigma U^{T})= U\Sigma (V^{T}V)\Sigma U^{T} = U \Sigma^{2} U^T$

* ***NOTE: This is highly inefficient and there are many more accurate way of computing these values***
## Matrix approximation

##### Recap
$U$ -- column space information 
$\Sigma^2$ -- Eigen information
$V^T$ -- row space information
All of them are also arranged in hierarchal order where $\sigma_{1} > \sigma_{2}$ or $U_{1} > U_{2}$ 

The first column $X_{1} = \sigma_{1}U_{1}V_{1}^{T}+ \sigma_{2}U_{2}V_{2}^{T}\dots+\dots\sigma_{m}U_{m}V_{m}^{T}+0$
$$X_{1,1} = \begin{bmatrix}
\vdots \\
U_{1} \\
\vdots  \\
\end{bmatrix}
\begin{bmatrix}
\dots\sigma_{1} \dots  \\
\end{bmatrix}
\begin{bmatrix}
\dots V_{1}^{T}\dots \\
\end{bmatrix}$$

- Event though $U$ is $n \times n$ , only $m$ columns are considered bc $\Sigma$ has $m$ unique non-zero values... based on this 

***Economy SVD***:
$X = U\Sigma V^{T} = \hat{U}\hat{\Sigma}V^T$ where $\hat{U}$ and $\hat{\Sigma}$ are $n \times m$ which is more efficient for $n \gg m$.
* Ex: Few examples of high resolution images where each image is a column

You can make it even ***more efficient*** by taking the rank $r$ of the matrix and truncating the approximation at $r$
$X_{1} = \sigma_{1}U_{1}V_{1}^{T}+ \sigma_{2}U_{2}V_{2}^{T}\dots+\dots\sigma_{r}U_{r}V_{r}^{T}$ 
Where you will only have to consider till 
$$X_{r,r} = \begin{bmatrix}
\vdots \\
U_{r} \\
\vdots  \\
\end{bmatrix}
\begin{bmatrix}
\dots\sigma_{r} \dots  \\
\end{bmatrix}
\begin{bmatrix}
\dots V_{r}^{T}\dots \\
\end{bmatrix}$$
Consider only first $r$ columns in $U$, first $r$ rows in $\Sigma$ and $V^{T}$
$X = U\Sigma V \approx \tilde{U} \tilde{\Sigma} \tilde{V^T}$ which was proved by [Eckart-Young Theorem](https://www.ime.usp.br/~jstern/miscellanea/seminario/Golub87.pdf) which proves that $$\text{argmin}\mid\mid X - \tilde{X}\mid\mid = \tilde{U}\tilde{\Sigma}\tilde{V^{T}}$$
 which says that the best approximation of $X$ which minimises the residuals with rank $r$ is  $\tilde{X}$  or $\tilde{U}\tilde{\Sigma}\tilde{V^{T}}$
 * *KEEP IN MIND:* $\tilde{U}\tilde{U^{T}} \neq I$

## Principal Component Analysis (PCA)

It is the bedrock for dimensionality reduction technique for *probability and statistics* and used a lot in *Machine Learning and Data Science* 

##### Brief
PCA, or Principal Component Analysis, is a statistical technique used for dimensionality reduction. It transforms a dataset into a new set of variables, called principal components, which capture the most important information and reduce the complexity of the data. The principal components are linear combinations of the original variables and are ordered by their ability to explain the data's variability. PCA helps in visualizing and analyzing high-dimensional data, identifying patterns, and highlighting the most influential features.

* We are going to be learning the ***statistical interpretation of SVD*** which is hierarchical coordinate system (based on data)

Instead of the one mentioned above, we are using a matrix where each row is an example
$$X = \begin{bmatrix}
\dots X_{1} \dots \\
\dots X_{2} \dots \\
\vdots \\
\dots X_{m} \dots
\end{bmatrix}$$

##### Steps:
1. Compute mean of every row to create a matrix
$$\bar{x} = \frac{1}{n} \sum_{j=1}^{n}y_{j}$$
$$\bar{X} = \begin{bmatrix}
1 \\
1 \\
1  \\
\vdots
\end{bmatrix} \begin{bmatrix}
\bar{x_{1}}  & \bar{x_{2}}  & \dots
\end{bmatrix}$$
2. Subtract mean from normal data to centre the data around 0
$B = X - \bar{X}$ and we can treat it as a Gaussian data with mean 0; also $B = U\Sigma V^T$

3. Covariance matrix (correlation matrix from SVD above) $C = BB^{T}$

4. Compute eigen values of $C$ 
$$C = VDV^{T} \text{ where V: eigen vectors \& D: eigen values}$$
5. We can compute ***principal components*** by using eigen vectors of $C$
$T \text{(Princ. Comp.)} = BV\text{(loading)}$

$\implies T = BV = U\Sigma VV^{T}= U\Sigma$  where $\mid\mid T\mid\mid \leq \mid \mid X\mid\mid$
* So if we want to know the variance being captured in the first $r$ rows then we can use $U\Sigma$ or $BV$ to know about the data from smaller matrix

$$\frac{\sum_{k=1}^{r}\lambda_{k}}{\sum_{k=1}^{n}\lambda_{k}} \text{ where } \lambda \text{ (eig.val)} = \sigma^2$$

We can compute this in MATLAB
```MATLAB
[V,score,s2] = pca(B)
```

* this how you end up extracting the feature vectors of the data where each row is 

##### **Side note: why correlation or covariance matrix works:***

You basically end up $X_{i}\times X_{j}$ for each row / column
$$\begin{bmatrix}
X_{1}^{T}X_{1} & X_{1}^{T}X_{2}  & \dots \\
X_{2}^{T}X_{1} & X_{2}^TX_{2} & \dots \\
\vdots & \vdots & \vdots
\end{bmatrix}$$

which is also subtracted from its means so it ends being

$$\begin{bmatrix}
\dots  &  \dots & \dots\\
\dots  & \sum(x_{i,i} - \bar{x})^{2} & \dots \\ \\
 & \text{ bc } X  = X^{T}\text{ for when i = j} &  \\
\sum(x_{i,j} - \bar{x})(x_{j,i} - \bar{x}^T) \\
 \text{same as diagonal but for } x \text{ \& }X^T & &   \\
\dots & \dots & \dots

\end{bmatrix}$$

This would form 

$$\begin{bmatrix}
Var(x_{1}) & \dots & Cov(x_{n},x_{1}) \\
\vdots & . & \vdots \\
Cov(x_{n},x_{1})  & \dots & Var(x_{n})
\end{bmatrix}$$

so ... you can have covariance matrix of a data by $N = X - \bar{X}$ and then $NN^{T}$

## Robust PCA

##### Motivation

Using Robust statistics, you can even identify a person, even if they are using fake moustache and glasses to trick you.

So, if you have a regression of data, it can be highly skewed `(7,-1)`


```chart
type: line
labels: [0,1,2,3,4,5,6,7]
series:
  - title: 
    data: [0,1,3,3,5,5,6,-1]
tension: 0.2
width: 80%
labelColors: false
fill: false
beginAtZero: false
bestFit: false
bestFitTitle: undefined
bestFitNumber: 0
```

***SO*** : Standard Mean Sum Squares regression ***very sensitive to outliers***


HOWEVER: if you penalise the ***Absolute Value of the Errors*** instead of the sum, you get a lot better fit on the line as shown in the example $\parallel . \parallel_{1}$ rather than $\parallel.\parallel_{2}$ 

![[Pasted image 20230714144911.png]]

This is the idea behind RPCA:
![[Pasted image 20230714145240.png]]
Same idea of feature vectors not being manipulated easily can be applied to data of higher dimension such as the picture below:
![[Pasted image 20230714145511.png]]

##### Basic idea:

So we would collect a large set of data $X$ and split it into ***Eigenvectors*** $L$ and ***Sparse matrix***  $S$ (pic above) which would like the outliers

To formalise the decomposition even more so we want $$\text{ min } rank(L) + \parallel S \parallel_{0} \text{ subject to }  L+S=X$$ So, constraints are to minimise rank of $L$ and make $S$ sparse

##### Convex relaxation:

This becomes an ***Impossible*** problem cause their is even more so, we ***Satisfice our decomposition*** 
  $$\text{min} \parallel L \parallel_{*} + \lambda_{0}\parallel S \parallel_{1} \text{ subject to } L + S = X$$

So, we use proxies which are easier to work with. 
* Proxy for rank is the ***nuclear norm*** *(sum of singular values)*
* Proxy for sparsity is the ***one norm*** *(sum of absolute values)*  

