function [v, best, details] = seqV(D, P, psi, k, nClasses, RNG)

% Find the fold indices for each observation in the dataset
rng(RNG)
fold_indxs = crossvalind('Kfold', size(D, 1), k);

v = 0; % keeps the best AUC seen
best = 1; % keeps the best classifier seen
n = length(psi); % number of classifiers to try
details = zeros(n, 1); % keeps the AUC for each classifier

% for each classifier
for i = 1:n
    v_current = 0; % keeps the sum of AUCs for each cross validation
    
    % for each cross validation fold
    for j = 1:k
        test = (fold_indxs == j);
        train = ~test;
        rng(i * j + RNG);
        model = psi{i}(D(train, :), P(train)); % train
        pred = predict(model, D(test, :)); % test
        
        % some clssifiers output cell array which can be converted to numeric
        if iscell(pred)
            pred = cell2mat(pred) - '0';
        end
        
        % check that the prediction array is numeric
        if isnumeric(pred) == 0
            error('predict() does not return a numeric vector or cell array of strings with classifier(%d)', i)
        end
        
        % compute confusion matrix and get AUC for current fold
        M = ComputeConfusionMatrix(P(test), pred, nClasses);
        if nClasses < 3 % binary classification problem
            TP = M(1, 1);
            TN = M(2, 2);
            FP = M(1, 2);
            FN = M(2, 1);
            AUC = ComputeTwoClassAUC(TP, FN, FP, TN);
        else % multiclass classification problem
            AUC = ComputeMultiClassAUC(M, nClasses);
        end
        v_current = v_current + AUC;
    end
    
    % final AUC for current classifier is the average of all cross valids
    v_current = v_current / k;
    details(i) = v_current;
    
    % check if current classifie r is the best seen so far
    if v_current > v % update best AUC seen and index of best classifier
        v = v_current;
        best = i;
    end
end

end