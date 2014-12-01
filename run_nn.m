load labeled_images.mat;
load public_test_images.mat;
%load hidden_test_images.mat;

if ~exist('hidden_test_images', 'var')
  test_images = public_test_images;
else
  test_images = cat(3, public_test_images, hidden_test_images);
end


%% Preprocessing the data
% USE_GABOR = True:     Training by extracting Gabor features from images
% USE_GABOR = False:    Training by learning pixel intensities
USE_GABOR = true;
if USE_GABOR
    addpath gabor
    inputs_train = gabor_features(tr_images);
    inputs_test = gabor_features(test_images);
    rmpath gabor
else
    num_training = size(tr_images, 3);
    inputs_train = reshape(tr_images, [1024, num_training]);
    num_testing = size(test_images, 3);
    inputs_test = reshape(test_images, [1024, num_testing]);
end


%% Cross validation and prediction
addpath neural_net
for K=[10 15 20 35 50]
  nfold = 10;
  %acc(K) = nn_cross_validate(K, inputs_train, tr_labels, nfold);
  [~, acc(K)] = nn_classifier(K, inputs_train, tr_labels, inputs_test);
  fprintf('%d-fold cross-validation with K=%d resulted in %.4f accuracy\n', nfold, K, acc(K));
end
[maxacc, bestK] = max(acc);
fprintf('K is selected to be %d.\n', bestK);

%bestK = 35;

% Run the classifier
prediction = nn_classifier(bestK, inputs_train, tr_labels, inputs_test);
rmpath neural_net


%% Print results

% Fill in the test labels with 0 if necessary
if (length(prediction) < 1253)
  prediction = [prediction'; zeros(1253-length(prediction), 1)];
end

% Print the predictions to file
fprintf('writing the output to nn_prediction.csv\n');
fid = fopen('nn_prediction.csv', 'w');
fprintf(fid,'Id,Prediction\n');
for i=1:length(prediction)
  fprintf(fid, '%d,%d\n', i, prediction(i));
end
fclose(fid);