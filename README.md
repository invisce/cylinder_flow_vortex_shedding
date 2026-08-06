# Flow Past a Cylinder (2D Incompressible Navier-Stokes)

A finite-difference solver for channel flow past a circular cylinder, built entirely from scratch in Python. This project simulates vortex shedding, famously known as the Karman vortex street. I built this to strengthen my engineering portfolio and show my understanding of numerical methods. 

## Overview
Flow past a cylinder is one of the most famous benchmark problems in Computational Fluid Dynamics (CFD). When a fluid flows around a blunt object like a cylinder, it creates a wake. At low speeds, the wake is calm. But as the speed increases, the wake becomes unstable and starts throwing off alternating swirling whirlpools this is the Karman vortex street. This project builds the math from the very beginning to simulate exactly how and why that happens.

## Governing Equations
The simulation is powered by the incompressible Navier-Stokes equations. Here is the math written out:

**Continuity (incompressibility):**

$$ \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0 $$

**Momentum (x and y directions):**

$$ \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial x} + \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) $$

$$ \frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial y} + \nu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right) $$

## Numerical Method
I programmed a Chorin-style projection method to step the simulation forward in time. Here is a breakdown of the process:

1. **Pressure-Poisson Equation:** In every single time step, the code calculates an intermediate velocity, and then solves a pressure-Poisson equation. This pulls the velocity back to being completely incompressible.
2. **Central Differencing:** For the convective terms, I used central differences. This is a very important choice. A simpler first-order upwind method adds a lot of fake "numerical diffusion". That fake drag acts like extra viscosity, which drops the effective Reynolds number too low and completely kills the vortex shedding. By using central differences, there is almost zero numerical diffusion, meaning the instability survives.
3. **Immersed Cylinder:** Modeling curved shapes on a square grid is hard. I treated the cylinder as an "immersed solid". I apply a circular mask that just forces the velocity to zero inside the cylinder area after every single update. It is a simple but very strong way to handle the boundary.

**Boundary Conditions:**

| Boundary | Condition |
|---|---|
| Inlet (left side) | $u = U_\infty, \ v = 0$ (constant incoming flow) |
| Outlet (right side) | Zero-gradient (the flow exits freely without bouncing back) |
| Top & Bottom walls | No-slip condition ($u = 0, \ v = 0$) |
| Cylinder Surface | No-slip condition (handled using the immersed mask) |

## Simulation Runs
The solver is set up to run two main tests to see how the fluid changes:

* **Re_D = 25:** This is a low-speed run. The wake stays completely steady and symmetric. It creates a pair of standing recirculation bubbles glued right behind the cylinder. Nothing sheds.
* **Re_D = 100:** This is a higher-speed run. The wake becomes unstable and starts shedding the Karman vortex street. A perfectly symmetric math equation needs a physical trigger to start shedding, so I added a tiny, brief jolt to the transverse velocity just behind the cylinder during the first half time-unit to cause the instability.

## Validation
To prove the Re_D = 100 run actually works correctly and simulates real physics, I verified it against the Strouhal number (St). 

$$ St = \frac{f \, D}{U_\infty} $$

For a cylinder wake in this Reynolds number range, the real-world experimental value should be approximately $St \approx 0.2$. I placed a virtual probe in the wake to track the transverse velocity over time. I found the shedding frequency ($f$) using two different methods:
1. Counting the zero-crossings of the velocity signal.
2. Running a Fast Fourier Transform (FFT).

Both methods gave a result near 0.2, which proves that the simulation oscillates at the exact right frequency, not just a random one.

## Requirements
To run this on your machine, you will need the following Python packages:
* `numpy`
* `matplotlib`
* `numba`

*Note:* `numba` is highly recommended. Using the `@njit` decorator with `fastmath=True` makes the heavy loops run in about a minute. If you run this with pure `numpy`, it will take about 4 minutes to finish.

## Running the Solver
Just load the script into Jupyter Notebook and run it. The code has a `report_stability` function that checks the grid parameters and time steps to make sure they stay safely under both the convective (CFL) and diffusive stability limits before it starts crunching numbers.

## Author 
Ali ghazvine
