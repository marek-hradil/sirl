# Optimal Control

## Learning objectives

* Describe what comprises an optimal control problem and the initial value problem of dynamical system constraints. 

<!--
* Briefly explain the general idea of direct method, calculus of variation and dynamic programming. (the later two might not be that useful I think. it is just for completeness and we can show why a PD design can make sense);

* Implement trajectory optimization with direct shooting and autograd to solve simple problems, such as Brachistochrone problem.

* Implement policy optimization with PD control to solve simple problems, such as Cart-pole.
-->
* Explain how an optimal control problem can be used to generate the desired motion for an articulated link to track a desired pose;

* Account for how numerical methods work to solve an optimal control problem as trajectory optimisation. Explain its benefits and potential disadvantages, e.g. sample-based (shooting) vs gradient-based. 

* Explain the difference of representing actions with trajectory or control policy such as PD control; 

* Implement trajectory optimization with autodiff and three-link articulated body model for a reaching task, e.g. finding a motion trajectory to let tip follow a goal trajectory, e.g. a circular curve; 

Possible assignment/exercise activities:

Implement optimal control of the three-link model. You may use your dynamics model in previous assignment or an off-shelf simulation, e.g. MuJoCo/tiny/Brax, to compute forward dynamics and gradients. Use the single shooting method and optimization method from scipy to obtain optimal trajectory and PD controller to move the mechanism from one configuration to another. Compare and provide possible intepretations to discuss the obtained results.

* Try to change the mechanism mass, e.g. setting a slightly different mass, see how will the resultant trajectories be like  when models are not exact;

* Try to change the simulator setup, e.g. integrator step, see how will the derived trajectory/controller be like under a different simulation setup; 

* Try to change the horizon of optimal control that account for, see how will this impact the computational cost and performance of derived trajectories and controller;



```
**Dependencies on previous assignment**:

Autodiff through the arm model dynamics or simulator to run trajectory optimization.
````

Test citations {cite:p}`bertsekas2019reinforcement` {cite:p}`Betts2010` {cite:p}

## References
```{bibliography}
:filter: docname in docnames
```