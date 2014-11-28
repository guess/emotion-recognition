clear all

% load labeled_images.mat
% 
% ntr = size(tr_images, 3);
% inputs_train = reshape(tr_images, [1024, ntr]);
% targets_train = full(ind2vec(tr_labels'));
% 
% c = 0.2255; % Best value we have so far. Note: It's incorrect %.
% hidden_u = [65, 75, 85, 95]; % Array of hidden units we want to test.
% h = 1; % Iterator
% while c >= 0.2255 && h < size(hidden_u,2) + 1
% 	net = patternnet(hidden_u(h));
% 	[net,tr] = train(net, double(inputs_train), double(targets_train));
% 	testX = inputs_train(:,tr.testInd);
% 	testT = targets_train(:,tr.testInd);
% 	testY = net(testX);
% 	%predictions = vec2ind(testY);
% 	[c,cm] = confusion(testT,testY);
%     fprintf('Number of Hidden Units: %d\n', hidden_u(h));
% 	fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
% 	fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);
%     h = h + 1;
% end
% 
% % public test data
% load public_test_images.mat
% ntest = size(public_test_images, 3);
% inputs_test = reshape(public_test_images, [1024, ntest]);
% testY = net(inputs_test);
% prediction = vec2ind(testY);

load labeled_images.mat;
load public_test_images.mat;

if ~exist('hidden_test_images', 'var')
  test_images = public_test_images;
else
  test_images = cat(3, public_test_images, hidden_test_images);
end

for K=[10 15 20 35 50]
  nfold = 10;
  acc(K) = nn_cross_validate(K, tr_images, tr_labels, nfold);
  fprintf('%d-fold cross-validation with K=%d resulted in %.4f accuracy\n', nfold, K, acc(K));
end
[maxacc, bestK] = max(acc);
fprintf('K is selected to be %d.\n', bestK);

% Run the classifier
prediction = nn_classifier(bestK, tr_images, tr_labels, test_images);

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