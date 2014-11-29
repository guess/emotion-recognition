load labeled_images.mat
load public_test_images.mat

prediction = svm_classifier(tr_images, tr_labels, public_test_images);
prediction = [prediction; zeros(1253-length(prediction), 1)];

fid = fopen('svm_prediction.csv', 'w');
fprintf(fid,'Id,Prediction\n');
for i=1:length(prediction)
    fprintf(fid, '%d,%d\n', i, prediction(i));
end
fclose(fid);