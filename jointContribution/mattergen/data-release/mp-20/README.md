The MP-20 dataset was first published by [Jain et al., 2013](https://pubs.aip.org/aip/apm/article/1/1/011002/119685):

```
Jain, A., Ong, S. P., Hautier, G., Chen, W., Richards, W. D., Dacek, S., ... & Persson, K. A. (2013). Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. APL materials, 1(1).
```

The MP-20 dataset is published under the [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

We applied the following modifications to the data:
* Exclude structures containing the elements `Tc`, `Pm`, or any element with atomic number 84 or higher.
* Relax structures with DFT using a PBE functional in order to have consistent energies.
* For the training set, remove any structure with energy above the hull higher than 0.1 eV/atom.