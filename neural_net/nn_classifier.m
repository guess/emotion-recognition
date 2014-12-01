function [test_prediction, c] = nn_classifier(num_hiddens, tr_images, targets_train, test_images)

% ntrain = size(tr_images, 3);
% inputs_train = reshape(tr_images, [1024, ntrain]);

targets_train = full(ind2vec(targets_train'));

if nargin < 4
    ntest = 0;
else
    ntest = 1;
end
    
net = patternnet(num_hiddens);
[net,tr] = train(net, double(tr_images), double(targets_train));

testX = tr_images(:,tr.testInd);
testT = targets_train(:,tr.testInd);
testY = net(testX);
test_prediction = vec2ind(testY);
[c,cm] = confusion(testT,testY);
c = 100*(1-c); % percentage of correct classification
fprintf('Number of Hidden Units: %d\n', num_hiddens);
fprintf('Percentage Correct Classification   : %f%%\n', c);
fprintf('Percentage Incorrect Classification : %f%%\n', 100-c);


if ntest > 0
    test_results = net(test_images);
    test_prediction = vec2ind(test_results);
end