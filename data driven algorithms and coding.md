
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



## Intro to modelling with matrices and vectors

## Sparse non-linear dynamics

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