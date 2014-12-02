function [CVMdl] = svm_classifier(tr_images, tr_labels, test_images)

rng(1); % For reproducibility


% Train an ECOC model using SVM binary classifiers 
t = templateSVM('Standardize', 1);
Mdl = fitcecoc(tr_images', tr_labels, 'Learners', t);
CVMdl = crossval(Mdl);
loss = kfoldLoss(CVMdl);
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-loss));
CMdl = CVMdl.Trained{3};           % Extract trained, compact classifier
test_prediction = predict(CMdl, test_images');
