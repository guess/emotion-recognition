%load labeled_images.mat;
load gabor_features.mat;
% load public_test_images.mat;
% load hidden_test_images.mat;

% if ~exist('hidden_test_images', 'var')
%   test_images = public_test_images;
% else
%   test_images = cat(3, public_test_images, hidden_test_images);
% end



%% Make prediction
mdl = fitcknn(inputs_train', tr_labels);
CVMdl = crossval(mdl);
loss = kfoldLoss(CVMdl);
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-loss));
CMdl = CVMdl.Trained{1};           % Extract trained, compact classifier
prediction = predict(CMdl, inputs_test');


%% Print results

% Fill in the test labels with 0 if necessary
if (length(prediction) < 1253)
  prediction = [prediction; zeros(1253-length(prediction), 1)];
end

% Save results to file
fid = fopen('knn_prediction.csv', 'w');
fprintf(fid,'Id,Prediction\n');
for i=1:length(prediction)
    fprintf(fid, '%d,%d\n', i, prediction(i));
end
fclose(fid);