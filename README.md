# Kicked-Rotor

This is a collection of programs developed during a 2 weeks internship at the University of Lille. 
The internship takes place in the Cold Atoms team of the PHLAM laboratory, "Quantum Chaos" group. This group has seven 
permanent members and four PhD students. The activities of the quantum chaos group are based on an internationally 
recognised expertise in the theoretical and experimental study of Anderson localization, a problem posed in the context 
of disordered media in condensed matter and addressed here in an original system based on cold atoms and the "stricken 
rotator" model. This group, which has several very fruitful collaborations with theorists and mathematicians through the 
CEMPI Laboratory of Excellence (European Centre for Mathematics, Physics and their Interactions) and with the Kastler 
Brossel laboratory (ENS, Sorbonne and Collège de France), which has three Nobel Prize winners.

This internship focused on the modelisation of the Kicked Rotor, both in classical and quantum cases, with the goal to study the specific case of a spin-1/2 particle.

***Disclaimer (06/2023):*** _This project was created before I knew how to use git, and before I had access to Pycharm. The structure
of the code might not be very intelligible. Moreover, the code might not be legible all the time with non-explicit variable
names. A short restructuring will be done over the time to make this project more accessible._

# Files structure

* `Classical` : Custom classes and methods to model the classical kicked rotor.
  * `classicalKickedRotor.py` : computed Chirikov standard map, as well as the energy levels
  * `classicalPlot.py` : utils methods to plot the probability distributions and phase diagram of the classical kicked rotor
  * `Graphs` : example of graphs obtained
* `Quantum` : Custom classes and methods to model the quantum kicked rotor.
  * `fermionsKickedRotor.py` : short script to take into account Fermi distribution 
  * `quantumKickedRotor.py` : class to model the quantum kicked rotor (no spin)
  * `symplecticKickedRotor.py` : class to model the spin-orbit interaction in the quantum kicked rotor
  * `Graphs` : series of graphs obtained for each of the three cases listed above. _The `Spin` folder also contained Gaussian
fitting method_
  * `LevelSpacing` : `levelspacing.py` computes the energy level distribution for three classes of symmetry of the Hamiltonian