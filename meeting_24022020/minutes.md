I discussed the basics of our HMM implementation. 
For reference, here are the published papers on this work:

Method:
https://doi.org/10.1098/rsta.2018.0150

First applications:
https://doi.org/10.1002/adts.201800168

Our goal is to reduce the number of quaderature points (QP) that require a molecular dynamcis simulation.

Each QP is characterised by its strain matrix (6 unique elements) and this matrix's history, which we currently approximate with a 10 point spline of the history of each element. In this way, each QP is characterised by 60 values. What we refer here as the "history of the strain matrix" can be seen as a trajectory within which each position is defined by 6 scalars.

The difference between two quadrature points is the absolute difference between every element of a QP's strain history, this is basically the L2 norm. In other words, we compute the L2-norm of the difference between the splines fitted to the two strain histories (square root of the sum of the differences between the 10 pairs of control points of the splines).

<img src="https://render.githubusercontent.com/render/math?math=\rm{Difference} = | \epsilon_{.,.,t}^i - \epsilon_{.,.,t}^j |">

Where <img src="https://render.githubusercontent.com/render/math?math=\epsilon_{.,.,t}^i"> , is every element of the strain history for QP i. 

Our current approach is to find QPs whose similarity is below a threshold value, call it **Gamma**. This threshold is currently set such as to find a comprise between walltime and *global accuracy* reduction. *Global accuracy* is assessed in comparison to a reference simulation without the use of the spline-based reduction algorithm. The choice of the threshold **Gamma** might not hold from one simulation configuration to another, therefore, quite annoyingly, its value should theoretically be verified for every new testing campaign. Hence, requiring to compute the whole system at least once without the spline-based algorithm.

MD sims for QPs to be ran are chosen based on a graph importance algorithm, which I explained during the meeting. In more details, a graph is setup to ensure we obtain the stress tensor for all QPs computing the **least** MD simulations. The design of the graph is not so trivial, it attributes importance to QPs proportionally to the number of other QPs it is similar too.

We should calculate an error for how well the chosen subset of QPs represents all QPs. 
This is the sum of difference between each QP and the QP representing it.
This error could be trivially minimized by adding more representative QPs (and increaseing computational cost), 
so instead we optimise a new term, call it d, where d = error in clustering + cpu penalty.
We could cluster QPs using a K-means algorithm, for each cluster we choose to **simulate the QP closest to the centroid** for that cluster.
Discussed a Kt-means algorithm which can choose a variable number of clusters to generate. 
The number of clusters could be increased to reduce the clustering error, so we add a penalty based on the computational cost of doing more MD simulations.

We can then formalise the overall error in our MD reduction strategy by using importance sampling. 
We can estimate the difference between simulating the entire set of QPs and a sub set of QPs. 
Benjamin could you add something here, maybe some reading I can do to understand this better.

We also talked about Voronoi partitions, and adaptive meshing in FEM. 
I think these are similar techniques for importance sampling, and definitely something we want to include further down the line.
