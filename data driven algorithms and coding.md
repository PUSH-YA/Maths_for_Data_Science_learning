 
## Data driven dynamical systems overview

We will be using machine learning and regression to solve such dynamical systems.

Time dependent ones are called ***non-autonomous*** and the others are called ***autonomous***

*Anatomy of a  dynamical system:*

$$\dot{x} = f(x,t,u;\beta)$$
where $x$ is state of system, $t$ is time, $u$ is activation and $\beta$ are parameters. The $f()$ determines the dynamical system. 

We can often not write such systems (climate, physics, finances, neurology etc.) through first principles.

* We went from First principles  -> Data driven systems (Kepler would be proud)

***Challenges (according to Steve Brunton):***
1. These are non-linear systems
2. Unknown dynamics and we need to discover $f()$
3. High dimensional, $\parallel \dot{x} \parallel = 1,000,000+$
4. Chaos and transient
5. Noise, stochastically forced 
	1. have to take stochasticity /unpredictable nature into account
6. Multiscale dynamics
	1. You can zoom in and in and it still has structure
7. Uncertainty in my initial conditions, parameters and so on

***Uses (according to Steve Brunton):***
1. future state prediction 
	1. might want to run for 1 trajectory or for an ensemble of targets (the Prob Density Function)
2. design and optimisation (F1, yachts, etc.)
3. modify and control the behaviour of the system actively
4. interpretability and physical intuition of such systems
	1. usually ML is not interpretable so we can tailor it to give us a more interpretable description of the dynamical system using ML

***Techniques:***
1. linear and sparse regression
2. neural networks and deep learning
3. Genetic algorithms and genetic programming

If we have certain constraints on the system such as the mass or energy being conserved then it makes the data driven dynamical systems a lot nicer and smarter

* Koopman analysis (touched it in [[Learn Neural networks]]) which increases dimensionality of the data to find a new coordinate system which linearizes the data 
$$f(x) \to \frac{d}{dt}\phi(x) = \lambda \phi(x)$$



## Intro to modelling with matrices and vectors  (A probabilistic weather model)

We are going to be modelling weather and we are going to be using 3 states to simplistically model the weather: *cloudy*, *rainy* and *nice*

This is going to have discrete evolution in time. We are going to be modelling it in a deterministic manner where if I know the state of the weather today, I will know the state of the weather tomorrow

So the probabilistic state machine looks as following:
![[Probabilistic weather model]]

The A matrix is made up of the table of the state today against the probability of the states the next day

$$X_{k+1} = AX_{k} = \begin{bmatrix}
0.5 & 0.5 & 0.25 \\
0.25 & 0.5 & 0.25 \\
0.25 & 0 & 0.5 \\
\end{bmatrix} \begin{bmatrix}
\vdots \\
X_{today} \\
\vdots
\end{bmatrix}$$
which outputs the probability of the things tmrw

We can use code to model this, He shows an example in `MATLAB` but I perfer `python` so I shall use that
```python
import numpy as mp
from matplotlib import pyplot as plt

A = np.array([0.5,0.5,0.25],
			[0.25,0.5,0.25],
			[0.25,0,0.5])

X_today = [1,0,0]
X_tmrw = (A@X_today)


weather = np.zeros((50,3))

X_today = np.array([[1],[0],[0]])
for k in range(50):
    X_tmrw = A@X_today
    weather[k] = [i[0] for i in X_tmrw]
    # print(k)
    # print(X_tmrw)
    X_today = X_tmrw
plt.plot(weather)
plt.grid(True)
```
The above code will fill the weather matrix as following where as following:
$$\begin{bmatrix}
P(R, day_{1}) & P(N, day_{1})  & P(C, day_{1}) \\
P(R, day_{2}) & P(N, day_{2})  & P(C, day_{2}) \\
P(R, day_{3}) & P(N, day_{3})  & P(C, day_{3}) \\
\vdots & \vdots & \vdots
\end{bmatrix}$$


![[Pasted image 20230721223655.png]]
This shows how if we run the probabilites where all the states converge in the long run

* These system can get very complex with large data set and a lot more states

## Model predictive control (MPC)

##### What MPC does
you run forecasts of this model forward in time for different actuation strategies $u$ and optimise the control of $u$ over a short time period and then determine the immediate next control action based on that optimisation

basically:
*system output -> optimise and apply action -> system output -> ...*

![[MPC]]
So in the diagram above, you can see how the feedback loop works
We can also see the constrains set on the input and the output of the systems

* Optimisation is at the heart of MPC and full optimisation is run at every time step (need fast hardware for it) -- Linear fast, nonlinear slow

Sometimes people do linear parameter varying so instead of one parameter, they have a family of linear parameters
$$ \dot{x} = Ax + Bu \to \dot{x} = A(\mu)x + B(\mu)u$$
Now you are running linear optimisation on different matrices of A and B