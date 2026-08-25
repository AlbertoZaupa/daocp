# A Dual Active-Set Solver for OCPs

This repository implements a dual active-set solver for Quadratic Programs (QPs) with Optimal Control structure:

```math
\begin{aligned}
\underset{x, u}{\min} \quad
& \frac{1}{2} x_N^{\top} Q_N x_N + q_N^{\top} x_N \\
& \quad + \sum_{t=0}^{N-1} \left(
    \frac{1}{2}
    \begin{bmatrix} u_t \\ x_t \end{bmatrix}^{\top}
    \begin{bmatrix} R_t & S_t \\ S_t^{\top} & Q_t \end{bmatrix}
    \begin{bmatrix} u_t \\ x_t \end{bmatrix}
    + \begin{bmatrix} r_t \\ q_t \end{bmatrix}^{\top}
      \begin{bmatrix} u_t \\ x_t \end{bmatrix}
\right) \\
\text{subject to} \quad
& x_{t+1} = A_t x_t + B_t u_t + w_t,
  \quad t = 0, \ldots, N-1, \\
& \underline{u}_t \leq u_t \leq \overline{u}_t,
  \quad t = 0, \ldots, N-1, \\
& \underline{x}_t \leq x_t \leq \overline{x}_t,
  \quad t = 1, \ldots, N, \\
& \underline{c}_t \leq C_{u,t} u_t + C_{x,t} x_t \leq \overline{c}_t,
  \quad t = 0, \ldots, N-1, \\
& \underline{c}_N \leq C_{x,N} x_N \leq \overline{c}_N, \\
& x_0 \text{ given.}
\end{aligned}
```

The solver is implemented entirely in `src.c`, and it relies on the linear algebra backend `blasfeo`.

## Installation

If you haven't cloned the repository yet, run:

```sh
git clone https://github.com/AlbertoZaupa/daqp-ocp.git --recurse-submodules
```

Otherwise, make sure you clone the `blasfeo` source code by running:

```sh
git submodule update --init
```

Now you need to set the target CPU architecture for `blasfeo` to build against. A few examples:

- If you have an Apple Silicon chip: `ARMV8A_APPLE_M1`.
- If you have an Intel chip (unless quite old): `X64_INTEL_HASWELL`.

You may look into `blasfeo/Makefile.rule` to see the list of available targets.
Create a `config.mk` file in the top-level directory and write the line:

```makefile
BLASFEO_TARGET=YOUR_TARGET
```

Now you are ready to build the solver, as well as the Python wrapper to run the examples. Simply run `make` from the top-level directory (use `make clean` if you want to delete your installation).

## Running the examples

3 examples are currently available to test and time the solver:

- `examples/python/di.py`: standard double integrator regulation problem.
- `examples/python/atlas.py`: disturbance rejection for a humanoid robot model.
- `examples/python/quadruped.py`: problem involving a quadruped with a manipulator installed on its back.

First make sure that you install the required Python dependencies:

```sh
pip install -r examples/python/requirements.txt
```

Then you can just run:

```sh
python examples/python/di.py
```
