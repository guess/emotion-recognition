clear all

load labeled_images.mat

ntr = size(tr_images, 3);
inputs_train = reshape(tr_images, [1024, ntr]);
targets_train = full(ind2vec(tr_labels'));

hidden_units = [1:15,20,30,40,50];
iterations = 5;
results = zeros(iterations, size(hidden_units,2));
for h = 1:size(hidden_units,2)
	% neural network with some hidden units
	for i = 1:iterations
		net = patternnet(hidden_units(h));
		[net,tr] = train(net, double(inputs_train), double(targets_train));
		testX = inputs_train(:,tr.testInd);
		testT = targets_train(:,tr.testInd);
		testY = net(testX);
		%predictions = vec2ind(testY);
		[c,cm] = confusion(testT,testY);
		results(i,h) = c;
	end
	mean_results = mean(results);
	fprintf('Number of Hidden Units: %d\n', hidden_units(h));
	fprintf('Average Percentage Correct Classification   : %f%%\n', 100*(1-mean_results(h)));
	fprintf('Average Percentage Incorrect Classification : %f%%\n', 100*mean_results(h));
end

[best_average_val, best_h_index] = min(mean_results);
net = patternnet(hidden_units(best_h_index));
[net,tr] = train(net, double(inputs_train), double(targets_train));
testX = inputs_train(:,tr.testInd);
testT = targets_train(:,tr.testInd);
testY = net(testX);
%predictions = vec2ind(testY);
[c,cm] = confusion(testT,testY);
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

% public test data
load public_test_images.mat
ntest = size(public_test_images, 3);
inputs_test = reshape(public_test_images, [1024, ntest]);
testY = net(inputs_test);
prediction = vec2ind(testY);

% Fill in the test labels with 0 if necessary
if (length(prediction) < 1253)
  prediction = [prediction'; zeros(1253-length(prediction), 1)];
end

% Print the predictions to file
fprintf('writing the output to prediction.csv\n');
fid = fopen('prediction.csv', 'w');
fprintf(fid,'Id,Prediction\n');
for i=1:length(prediction)
  fprintf(fid, '%d,%d\n', i, prediction(i));
end
fclose(fid);