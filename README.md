# 2D Cylinder Flow: Incompressible Navier–Stokes Solver

A finite-difference project for flow past a circular cylinder, developed in Python. The repository retains the original exploratory notebook and adds a separate standalone solver in `main.py`. The standalone solver produces periodic vortex shedding at a cylinder Reynolds number of 100 and reports wake velocity, lift, drag, vorticity, and shedding frequency.

The standalone solver uses a staggered MAC grid: pressure is stored at cell centres, while velocity components are stored on cell faces. A pressure projection enforces discrete mass conservation at each time step. The cylinder is represented by blocked Cartesian cells, so its curved surface is approximated by a staircase.

## From notebook to standalone solver

The notebook remains in the repository as the earlier, self-contained implementation and explanation. `main.py` is a separate numerical implementation of the same flow problem, with these changes:

| Aspect | Original notebook | Standalone `main.py` |
| --- | --- | --- |
| Variable placement | Velocity and pressure arrays share grid locations. | Pressure is cell-centred; velocity components live on cell faces (MAC grid). |
| Pressure solve | Numba-accelerated red-black SOR iterations. | A sparse pressure system is factored once with SciPy and solved at each time step. |
| Mass conservation | Uses central differences for the pressure correction and velocity divergence. | Builds the pressure operator from the same face fluxes used to measure divergence and reports the resulting divergence. |
| Cylinder forces | Integrates pressure contributions on the masked boundary. | Adds a first-order estimate of wall shear to the pressure contribution; the staircase boundary still limits force accuracy. |
| Frequency analysis | Uses lift zero crossings and a raw FFT. | Uses interpolated zero crossings and a windowed FFT with peak interpolation; reports the analysis window and frequency resolution. |

The results below belong to `main.py`. They do not retroactively validate the notebook: the two implementations should be assessed with their own settings and convergence checks.

## Model and numerical method

The fluid is governed by the two-dimensional incompressible Navier–Stokes equations:

$$\nabla\cdot\mathbf{u}=0,\qquad
\frac{\partial\mathbf{u}}{\partial t}+\mathbf{u}\cdot\nabla\mathbf{u}
=-\frac{1}{\rho}\nabla p+\nu\nabla^2\mathbf{u}.$$

The implementation advances the convective and viscous terms explicitly, solves a pressure Poisson equation, and corrects the face velocities. The pressure operator and velocity correction share the same face stencil; this makes the corrected field discretely divergence-free, including next to blocked cylinder cells. The sparse pressure system is factored with SciPy and reused throughout the run.

The default case uses a uniform inlet velocity, no-slip channel walls, a stationary cylinder, and an outlet pressure reference. An optional slip-wall setting is also available. A small, one-time transverse disturbance downstream of the cylinder starts the asymmetric shedding mode.

| Parameter | Default value |
| --- | ---: |
| Cylinder Reynolds number, $Re_D=U_\infty D/\nu$ | 100 |
| Domain, $L_x\times L_y$ | $8\times2$ |
| Cylinder diameter and centre | $D=0.4$, $(x_c,y_c)=(2,1)$ |
| Blockage ratio, $D/L_y$ | 0.20 |
| Grid | $400\times100$ pressure cells |
| Time step and number of steps | $0.002$, 25,000 |

The stated grid corresponds to `nx=401, ny=101` grid vertices in the Python constructor.

## Results from the default case

Statistics below use the final 40% of a 50-time-unit run of `main.py` (time 30–50). They should be recalculated if the method or settings change.

| Diagnostic | Result |
| --- | ---: |
| Strouhal number from lift zero crossings | 0.2369 |
| Strouhal number from windowed lift FFT | 0.2367 |
| Maximum discrete velocity divergence in the analysis window | $2.53\times10^{-14}$ |
| Mean drag coefficient, staircase estimate | 1.9385 |
| Lift coefficient amplitude, staircase estimate | 0.5357 |

The two Strouhal estimates agree, but both come from the **same lift history**: their agreement is an internal consistency check, not independent validation against a reference solution. The FFT window spans 20 time units, giving a frequency-bin spacing of 0.05 inverse time units; the reported FFT peak is interpolated between bins. The large force transient immediately after initialization is excluded from the reported averages.

The low divergence measures how well the **discrete** continuity equation is satisfied. It does not, by itself, establish grid convergence or accuracy of the cylinder forces. For a confined cylinder, published frequencies must be compared using a matching blockage ratio and boundary conditions; an unconfined-cylinder Strouhal number is not a direct benchmark for this case. See [Sahin and Owens (2004)](https://doi.org/10.1063/1.1668285) for the effect of wall confinement.

## Run

Install `numpy`, `scipy`, and `matplotlib`. Install `pillow` as well to export the animation.

```bash
python -m pip install numpy scipy matplotlib pillow
python main.py
```

The default run writes `cfd_validation_dashboard.png` and `vortex_street.gif` to the current working directory. It also prints the pressure-solve residual, maximum discrete divergence, and post-transient frequency and force estimates. To change the grid, Reynolds number, time step, or wall condition, edit the solver construction and `solve()` call in `main.py`. A full default run uses 25,000 time steps and can take several minutes depending on the computer.

## Scope and next checks

- Run grid and time-step refinement studies for $St$, mean $C_D$, and lift amplitude.
- Compare against a published case with the same blockage, inlet profile, wall conditions, and force normalization.
- Refine the staircase force integration or use a more accurate representation of the curved boundary before treating drag and lift as benchmark-quality values.

**Author:** Ali Ghazvine
