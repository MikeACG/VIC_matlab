% See the README for an explanation of this example
load fisheriris
classifiers = {
    @(X, Y) fitclinear(X, Y, 'Learner', 'logistic')
    @(X, Y) fitcknn(X, Y, 'numNeighbors', 3)
    @(X, Y) TreeBagger(100, X, Y)
};
clust_alg = @(X) kmeans(X, 2);
[a, b, c] = vic(meas, classifiers, clust_alg, 5, 123, 1);
disp(a);
disp(b);
disp(c);