These tarballs contain the data written by 120 processes (out of 319) during an HMM simulation of a tension test on a cuboid sample. 

The processes deal with a subset of the quadrature points of the complete finite element mesh of the system. In turn, each of these 120 CSV files contain the mechanical state (strain and stress tensors) at each timestep for the quadrature points they are in charge of.

The tar.gz archives can be decompressed in a single directory executing:
```
for f in *.tar.gz; do tar -xzvf "$f"; done
```

The lines in these CSV files are structured in an identical way, as:
```
timestep,cell,qpoint,material,strain_00,strain_01,strain_02,strain_11,strain_12,strain_22,updstrain_00,updstrain_01,updstrain_02,updstrain_11,updstrain_12,updstrain_22,stress_00,stress_01,stress_02,stress_11,stress_12,stress_22
```
Note that the 'updstrain_*' values refer to data specific to our iterative solution algorithm and do not refer to anything physical.
