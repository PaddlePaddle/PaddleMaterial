This dataset contains structures from the Alexandria ([Schmidt et al. 2022](https://archive.materialscloud.org/record/2022.126)) and MP-20 datasets. For details on MP-20, see [here](../mp-20/README.md).

The Alexandria dataset was published under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode) by:
```
Jonathan Schmidt, Noah Hoffmann, Hai-Chen Wang, Pedro Borlido, Pedro J. M.A. Carriço, Tiago F. T. Cerqueira, Silvana Botti, Miguel A. L. Marques, Large-scale machine-learning-assisted exploration of the whole materials space, Materials Cloud Archive 2022.126 (2022), https://doi.org/10.24435/materialscloud:m7-50
```

We applied the following modifications to the data:
* Exclude structures containing the elements `Tc`, `Pm`, or any element with atomic number 84 or higher.
* Relax structures with DFT using a PBE functional in order to have consistent energies.
* For the training set, remove any structure with more than 20 atoms inside the unit cell.
* For the training set, remove any structure with energy above the hull higher than 0.1 eV/atom.