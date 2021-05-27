# Validity Index using supervised Classifiers (VIC) MATLAB implementation
VIC is a clustering validation technique that is designed to be independent of any specific distance function to evaluate compactness within objects in the same cluster or separation between objects of different clusters. Instead, it is proposed that the cluster memberships returned by a given clustering algorithm upon running on an input unlabeled dataset can be used as classification targets for that same dataset.

VIC performs clasification training and testing in a k-fold cross validation pipeline with as many classifiers as the user provides obtaining an average Area Under the Receiving Operator Characteristic Curve (AUROC or AUC for short) for each classifier. By looking at the maximum AUC obtained by any of the classifiers, it is possible to asses the quality of a partition given by the clustering algorithm when comparing against different runs of VIC for different clustering algorithms and/or different parameters of the same clustering algorithm. VIC is based around the idea that better clustering partitions should translate into more learnable structures in the data which reflects in high performance in at least one of a group of expert classifiers.

For more information on VIC and comparisons against other classic clustering validation metrics, please read the original publication by the authors of VIC:

Jorge Rodríguez, Miguel Angel Medina-Pérez, Andres Eduardo Gutierrez-Rodríguez, Raúl Monroy, Hugo Terashima-Marín, Cluster validation using an ensemble of supervised classifiers, *Knowledge-Based Systems* Volume 145, 2018, Pages 134-144, ISSN 0950-7051, [https://doi.org/10.1016/j.knosys.2018.01.010](https://doi.org/10.1016/j.knosys.2018.01.010).

## Usage
We will go through the contents of the `example.m` script in order to understand how to use this MATLAB implementation of VIC.

### Syntax
To run VIC use the following syntax (programmed and tested in MATLAB_R2019b):

`[a, b, c] = vic(D, psi, omega, k, RNG, cores)`

#### Inputs
- `D` is an unlabeled dataset of `n` objects (rows) by `m` features (columns) passed as either a numeric array or table.
- `psi` is a 1-dimensional cell array of classifiers. Each classifier must be represented by a function handle.
- `omega` is the clustering algorithm to be evaluated. Must be passed as a function handle.
- `k` is the number of folds to be used for k-fold cross validation of each classifier.
- `RNG` is the random number generator seed to be used for reproducibility of the cross validation folds and classifiers whose training involves generation of random numbers. This must be an integer with the restrictions imposed by MATLAB's `rng()` builtin function.
- `cores` the number of cores used to run the process. Note that to take advantage of multiple cores in your machine, you need to have the Parallel Computing Toolbox installed in your MATLAB. Either using `1` for this parameter or not having the toolbox installed will result in VIC running in sequential mode with no parallel processes. Also note that when using more than 1 core, there will be an overhead time for the Parallel Computing Toolbox to setup the parallel process, so using multiple cores will only be benefitial when for example, evaluating several clustering algorithms or clustering algorithm parameters inside a loop and calling VIC multiple times.

#### Outputs
- `a` the best AUC achieved by any of the classifiers.
- `b` (optional) the index in `psi` of the classifier that achieved the best AUC.
- `c` (optional) 1-dimensional array of the AUCs achieved by each classifier in the same order as provided in `psi`.

### Setup an example experiment
Load a simple dataset.

`load fisheriris`

The latter should load two variables to the MATLAB workspace: `meas` which are the features of each object and `species` which are the labels (we won't use them for anything in this example, in fact we will assume that we don't know them and that we want to see if our unsupervised classification of `meas` is good).

Setup logistic regression, k-nearest neighbors and Random Forest classifiers.

```matlab
classifiers = {
    @(X, Y) fitclinear(X, Y, 'Learner', 'logistic')
    @(X, Y) fitcknn(X, Y, 'numNeighbors', 3)
    @(X, Y) TreeBagger(100, X, Y)
};
```

Each classifier should be a function handle which takes only 2 arguments: `X` which inside VIC will be a training set (e.i. a subset of `meas`) and `Y` which inside VIC will be the cluster memberships of the objects in `X`. Each of these function handles call the classifier perse; `X` and `Y` should be passed in the correct positions of the classifier function as indicated by the documentation of the learning algorithm that you wish to use. The rest of the parameters of the classifiers will remain constant during the execution of VIC (e.g. here the number of neighbors 3 in the k-nearest neighbors classifier and the number of iterations 100 for Random Forest). This allows for the user to set the parameters of each classifier as they wish and allows them to use any classifiers they want. The only restriction regarding the use of a given classifier is that the model returned by the classifier function should be compatible with the `predict()` MATLAB builtin returning either a numeric or string cell array of predicted classes.

Setup the clustering algorithm.

`clust_alg = @(X) kmeans(X, 2);`

This is a single function handle which takes only 1 argument `X` that inside VIC will just be the full provided dataset. One can also customize the parameters of the algorithm in the function handle (for example here we are specifying kmeans with the desired number of clusters = 2). The clustering algorithm should return a 1-dimensional numeric array where the possible cluster memberships start at 1 and increase by one (as it is the case for kmeans in MATLAB). Note that if for example, the clustering algorithm that we want to evaluate does not support memberships starting from 1 and rather returns three possible cluster memberships -1, 0 and 1, one can easily make it a valid input for VIC by doing `clust_alg = @(X) clustering_process(X) + 2;`. Similar ideas can be applied to transform a non-numeric output array to numeric by modifying the function handle. This makes it possible to use any pre-programmed clustering algorithm as long as the output is parsed correctly in the function handle.

Finally run VIC (here we use a 5-fold cross validation, a random number seed 123 and 1 core to run the process).

```matlab
[a, b, c] = vic(meas, classifiers, clust_alg, 5, 123, 1);
disp(a); % best AUC
disp(b); % index of best classifier by AUC
disp(c); % AUCs for each classifier
```