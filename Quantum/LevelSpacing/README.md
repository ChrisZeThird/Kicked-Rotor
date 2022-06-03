## Level Spacing

A great way to identify the type of symmetry of a system is to plot its level spacing distribution. I start by calculating the operator U, I then compute its eigenvalues
and their arguments in order to get to the energies. The plots are then fit with the theoritical curves of the random matrix theory model.

## Kramers' Degeneracy

In the case of the symplectic, the spin coupling leads to a degeneracy of 2 on the eigenvalues of energy. Therefore, only half the values obtained are used
in the distribution to avoid problems (the large number of 0's distort the histogram). You can check this by yourself by not adding `[::2]` on the phi in the symplectic case.

## Two Spin-1/2 systems

The first system is based on the work of Smielansky and Sharf, that is to say a coupling on the kick. The second one comes from the Thesis of Tony Prat, where he 
decided to use the propagation operator for the coupling. Experimentaly, the second is much interesting since it doesn't require short times unlike the Smielansky's one.
Consequently, it represents a great interest for the study of the symplectic kicked rotor.
