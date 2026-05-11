# Articulated Body Dynamics

## Learning objectives

<!--
* Explain how articulated systems are represented by generalized coordinates and velocity. Show the relation between generalized velocity and the velocity and time derivative of coordinate of a link body;

* Or shall we start with spatial vectors?

* Briefly explain what is forward and inverse dynamics doing for an articulated system, and their roles in simulation and control;

* Derive dynamics of a single joint and link with Newton's law and the recursive relation between two linked bodies (multiple DOFs of joint?);

* Describe how articulated dynamics equation are structured, how recursive newton equation algorithms work in general to compute inverse dynamics and how composite rigid body algorithms work to compute forward dynamics;

* Implement code snippets based on dynamics repo/simulators to animate character and access dynamcis-related terms.
-->
* Explain the concept of generalized coordinate and velocity in representing the state of articulated bodies;
* Explain the structure of articulated dynamics equation and identify dependencies of each term on generalized coordinate and velocity;

* Explain what are forward and inverse dynamics problems and how are they related to simulation and control in the context of robotics;

* Describe how recursive algorithms work in computing inverse and forward dynamics;

* Implement recursive algorithms to simulate a fixed-based three-link articulated body with 1-DOF revolute joint given a Spatial Algebra library (using RAINBOW);


Possible assignment/exercise activities:

Read the document of a simulator, e.g. MuJoCo/tiny/Brax, and implement the above articulated body. Using the simulator API to access relevant dynamical terms and compare them with the results from RNEA. (Shall we also run forward dynamics and compare the simulated trajectory as well? This might involve integrator selection...)

Maybe a more practical alternative is to use a set of Lego links to construct random mechanisms and build articulated model for them.

Using Mujoco as something to alleviate the dependency.

```
**Dependencies on previous assignment**:

They probably need to implement the model with jax to eval grad for furture use.
````

Here I am trying to cite a few materials to test citation feature {cite:p}`macklinthesis` {cite:p}`todorov2012mujoco` {cite:p}`modernrobotics` {cite:p}`featherstone2007` 

## References
```{bibliography}
:filter: docname in docnames
```