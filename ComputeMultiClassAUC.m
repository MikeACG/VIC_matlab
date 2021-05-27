function AUC = ComputeMultiClassAUC(confusionMatrix, k)
% Based on implementation at
% https://github.com/miguelmedinaperez/DTAE/blob/master/core/AUCCalculatorExtensions.cs

TP = 0;
FN = 0;
FP = 0;
TN = 0;
for i = 1:k
    TP = TP + confusionMatrix(i, i);
    for j = 1:k
        if i ~= j
            FN = FN + confusionMatrix(i, j);
            FP = FP + confusionMatrix(j, i);
            TN = TN + confusionMatrix(j, j);
        end
    end
end
AUC = ComputeTwoClassAUC(TP, FN, FP, TN);
end