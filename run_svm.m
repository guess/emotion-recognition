load labeled_images.mat
load public_test_images.mat

ntr = size(tr_images, 3);
inputs_train = reshape(tr_images, [1024, ntr]);

ntest = size(test_images, 3);
inputs_test = reshape(test_images, [1024, ntest]);

prediction = svm_classifier(inputs_train, tr_labels, inputs_test);
prediction = [prediction; zeros(1253-length(prediction), 1)];

fid = fopen('svm_prediction.csv', 'w');
fprintf(fid,'Id,Prediction\n');
for i=1:length(prediction)
    fprintf(fid, '%d,%d\n', i, prediction(i));
end
fclose(fid);