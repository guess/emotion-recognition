load labeled_images.mat;
load public_test_images.mat;
load hidden_test_images.mat;
load gabor_features.mat;
load trainedModel;

if ~exist('hidden_test_images', 'var')
  test_images = public_test_images;
else
  test_images = cat(3, public_test_images, hidden_test_images);
end

%% Preprocessing the data
addpath gabor
if ~exist('inputs_train', 'var')
    inputs_train = gabor_features(tr_images);
    inputs_test = gabor_features(test_images);
    save('gabor_features', 'inputs_train', 'inputs_test', 'tr_labels');
end
rmpath gabor


%% Make prediction
addpath svm
if ~exist('svm_model', 'var')
    svm_model = svm_classifier(inputs_train, tr_labels);
    save('trainedModel', 'svm_model');
end
rmpath svm

prediction = predict(svm_model, inputs_test');

%% Print results

% Fill in the test labels with 0 if necessary
if (length(prediction) < 1253)
  prediction = [prediction; zeros(1253-length(prediction), 1)];
end

% Save results to file
fid = fopen('svm_prediction.csv', 'w');
fprintf(fid,'Id,Prediction\n');
for i=1:length(prediction)
    fprintf(fid, '%d,%d\n', i, prediction(i));
end
fclose(fid);