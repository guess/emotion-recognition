load gabor_features.mat

% Keep 1/3 of the data for the combiner
total_training = size(tr_labels, 1);
num_training_combiner = floor(total_training/5);
num_training = total_training - num_training_combiner;

combiner_inputs = double(inputs_train(:,1:num_training_combiner));
combiner_labels = double(tr_labels(1:num_training_combiner));
train_inputs = double(inputs_train(:,num_training_combiner+1:total_training));
train_labels = double(tr_labels(num_training_combiner+1:total_training));


%% Run different classifiers

%% Support Vector machine
t = templateSVM('Standardize', 1);
Mdl = fitcecoc(train_inputs', train_labels, 'Learners', t);
CVMdl = crossval(Mdl);
loss = kfoldLoss(CVMdl);
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-loss));
CMdl = CVMdl.Trained{1};           % Extract trained, compact classifier
svm_prediction = predict(CMdl, combiner_inputs');
test_svm_prediction = predict(CMdl, inputs_test');


%% Neural network
addpath neural_net
fprintf('Starting meural network classifier...\n');
for K=[10 15 20 35 50]
  nfold = 10;
  [~, acc(K)] = nn_classifier(K, train_inputs, train_labels, combiner_inputs);
  fprintf('%d-fold cross-validation with K=%d resulted in %.4f accuracy\n', nfold, K, acc(K));
end
[maxacc, bestK] = max(acc);
fprintf('K is selected to be %d.\n', bestK);
net_target = full(ind2vec(train_labels'));
rmpath neural_net
net = patternnet(bestK);
[net,tr] = train(net, double(train_inputs), double(net_target));
testX = train_inputs(:,tr.testInd);
testT = net_target(:,tr.testInd);
testY = net(testX);
[c,cm] = confusion(testT,testY);
c = 100*(1-c); % percentage of correct classification
fprintf('Percentage Correct Classification   : %f%%\n', c);
nn_prediction = vec2ind(net(combiner_inputs));
test_nn_prediction = vec2ind(net(inputs_test));

%% KNN classifier
fprintf('Starting KNN classifier...\n');
mdl = fitcknn(train_inputs', train_labels);
CVMdl = crossval(mdl);
loss = kfoldLoss(CVMdl);
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-loss));
CMdl = CVMdl.Trained{1};           % Extract trained, compact classifier
knn_prediction = predict(CMdl, combiner_inputs');
test_knn_prediction = predict(CMdl, inputs_test');


%% Combiner
predictions = [nn_prediction' svm_prediction];
test_predictions = [test_nn_prediction' test_svm_prediction];


% addpath svm
% fprintf('Starting combiner...');
% prediction = svm_classifier(predictions', combiner_labels, test_predictions');
% rmpath svm

mdl = fitNaiveBayes(predictions, combiner_labels);
prediction = predict(mdl, test_predictions');



%% Print results

% Fill in the test labels with 0 if necessary
if (length(prediction) < 1253)
  prediction = [prediction; zeros(1253-length(prediction), 1)];
end

% Save results to file
fid = fopen('combiner_prediction.csv', 'w');
fprintf(fid,'Id,Prediction\n');
for i=1:length(prediction)
    fprintf(fid, '%d,%d\n', i, prediction(i));
end
fclose(fid);