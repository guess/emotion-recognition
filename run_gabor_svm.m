load labeled_images.mat
load public_test_images.mat
load hidden_test_images.mat

nfeat = 2560;   % number of gabor features
ga = gaborFilterBank(5, 8, 39, 39);
test_images = cat(3, public_test_images, hidden_test_images);

% Extract Gabor features from the training data
ntrain = size(tr_images, 3);
feat_train = zeros(nfeat, ntrain);
for i=1:ntrain
    feat_train(:,i) = gaborFeatures(tr_images(:,:,i), ga, 4, 4);
end

% Extract Gabor features from the test data
ntest = size(test_images, 3);
feat_test = zeros(nfeat, ntest);
for i=1:ntest
    feat_test(:,i) = gaborFeatures(test_images(:,:,i), ga, 4, 4);
end

% Run SVM on the Gabor features
prediction = svm_classifier(feat_train, tr_labels, feat_test);
prediction = [prediction; zeros(1253-length(prediction), 1)];

fid = fopen('svm_gabor_prediction.csv', 'w');
fprintf(fid,'Id,Prediction\n');
for i=1:length(prediction)
    fprintf(fid, '%d,%d\n', i, prediction(i));
end
fclose(fid);