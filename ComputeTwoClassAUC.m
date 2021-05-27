function AUC = ComputeTwoClassAUC(TP, FN, FP, TN)
% Based on implementation at
% https://github.com/miguelmedinaperez/DTAE/blob/master/core/AUCCalculatorExtensions.cs

positives = TP + FN;
negatives = TN + FP;
if positives > 0
    tprate = TP / positives;
else
    tprate = 1;
end
if negatives > 0
    fprate = TN / negatives;
else
    fprate = 1;
end
AUC = (tprate + fprate) / 2;
end