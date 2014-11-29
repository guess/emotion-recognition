function [test_prediction] = svm_classifier(tr_images, tr_labels, test_images)

% Prepare input data
ntr = size(tr_images, 3);
inputs_train = reshape(tr_images, [1024, ntr]);

ntest = size(test_images, 3);
inputs_test = reshape(test_images, [1024, ntest]);

rng(1); % For reproducibility

% Train an ECOC model using SVM binary classifiers 
% Specify a 30% holdout sample
t = templateSVM('Standardize', 1);
Mdl = fitcecoc(double(inputs_train'), double(tr_labels), 'Learners', t);
%CVMdl = fitcecoc(double(inputs_train'), double(tr_labels), 'Holdout', 0.30, 'Learners', t);
CVMdl = crossval(Mdl);
loss = kfoldLoss(CVMdl);
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-loss));
CMdl = CVMdl.Trained{1};           % Extract trained, compact classifier

% testInds = test(CVMdl.Partition);  % Extract the test indices
% XTest = inputs_train(testInds,:);
% YTest = tr_labels(testInds,:);
% 
% test_prediction = predict(CMdl, XTest);
% idx = randsample(sum(testInds),10);
% table(YTest(idx),test_prediction(idx),...
%     'VariableNames',{'TrueLabels','PredictedLabels'})

test_prediction = predict(CMdl, double(inputs_test'));






