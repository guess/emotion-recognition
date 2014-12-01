% The main function
% If test_images is provided, it will predict the results for those too, otherwise predicts 0 for the test cases.

load labeled_images.mat;
load public_test_images.mat;
%load hidden_test_images.mat;

total = size(tr_images, 3);
nvalid = total/3;
ntr = total - nvalid;

%inputs_train = reshape(tr_images, [1024, ntr]);
inputs_train = reshape(tr_images(:,:,1:ntr), [1024, ntr]);
target_train = tr_labels(1:ntr,:);
target_train = target_train';

inputs_valid = reshape(tr_images(:,:,ntr+1:total), [1024, nvalid]);
target_valid = tr_labels(ntr+1:total,:);
target_valid = target_valid';

%h = size(tr_images,1);
%w = size(tr_images,2);

if ~exist('hidden_test_images', 'var')
  ntest = size(public_test_images, 3);
  inputs_test = reshape(public_test_images, [1024, ntest]);
else
  inputs_test = cat(3, public_test_images, hidden_test_images);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% initialize the net structure.
num_inputs = size(inputs_train, 1);
num_hiddens = 10;
num_outputs = 7;

%%% make random initial weights smaller, and include bias weights
W1 = 0.01 * randn(num_inputs, num_hiddens);
b1 = zeros(num_hiddens, 1);
W2 = 0.01 * randn(num_hiddens, num_outputs);
b2 = zeros(num_outputs, 1);

dW1 = zeros(size(W1));
dW2 = zeros(size(W2));
db1 = zeros(size(b1));
db2 = zeros(size(b2));

eps = 0.01;  %% the learning rate 
momentum = 0.25;   %% the momentum coefficient

num_epochs = 50; %% number of learning epochs (number of passes through the
                 %% training set) each time runbp is called.

total_epochs = 0; %% number of learning epochs so far. This is incremented 
                    %% by numEpochs each time runbp is called.

%%% For plotting learning curves:
min_epochs_per_plot = 200;
train_errors = zeros(1, min_epochs_per_plot);
valid_errors = zeros(1, min_epochs_per_plot);
epochs = [1 : min_epochs_per_plot];

% Make everything a double 
W1 = double(W1);
W2 = double(W2);
inputs_train = double(inputs_train);
inputs_valid = double(inputs_valid);

