function [test_prediction] = svm_classifier(tr_images, tr_labels, test_images)

% Prepare input data
% ntr = size(tr_images, 3);
% inputs_train = reshape(tr_images, [1024, ntr]);
% 
% ntest = size(test_images, 3);
% inputs_test = reshape(test_images, [1024, ntest]);

rng(1); % For reproducibility

% Train an ECOC model using SVM binary classifiers 
%t_knn = templateKNN('Standardize', 1);
t = templateSVM('Standardize', 1);
Mdl = fitcecoc(double(tr_images'), double(tr_labels), 'Learners', t);
%CVMdl = fitcecoc(double(inputs_train'), double(tr_labels), 'Holdout', 0.30, 'Learners', t);
CVMdl = crossval(Mdl);
loss = kfoldLoss(CVMdl);
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-loss));
CMdl = CVMdl.Trained{1};           % Extract trained, compact classifier

test_prediction = predict(CMdl, double(test_images'));






