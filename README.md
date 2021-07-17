# Pseudo-Spectral 1d RK4 Scheme
The algorithm implemented in `main.cpp` can be used to perform a numerical integration of a spatially one-dimensional partial differential equation. This simulation uses a pseudo-spectral approach meaning that the equation is solved in Fourier space while nonlinear contributions are calculated by transforming back and forth between real and Fourier space. The boundary conditions are therefore periodic. The Fourier transformations involved here are executed by means of the fftw3 library. Adjust the global parameters `N`, `dtime`, and `systemsize` to configure the discretization settings and the function `pde(..)` according to the pde at interest. The present demo implements the KdV equation.

Compile using for example
   ```console
   $ g++ main.cpp -o run -lfftw3 -std=c++11
   ```
for the use of the GNU compiler. The output during execution will be written to the `data/` subdirectory.