function [CVMdl] = ensemble(tr_images, tr_labels, test_images)

% Prepare input data
ntr = size(tr_images, 3);
inputs_train = reshape(tr_images, [1024, ntr]);

ntest = size(test_images, 3);
inputs_test = reshape(test_images, [1024, ntest]);

rng(1); % For reproducibility

% Train an ECOC model using SVM binary classifiers 
%CVMdl = fitensemble(double(inputs_train'), double(tr_labels), 'Subspace', 15, 'KNN');

%t = templateKNN('NumNeighbors',5,'Standardize',1);
CVMdl = fitensemble(double(inputs_train'),double(tr_labels),'AdaBoostM2',50,'Tree');
%tEnsemble = templateEnsemble('Subspace', 100, t);

%pool = parpool; % Invoke workers
%options = statset('UseParallel',1);
%Mdl = fitcecoc(double(inputs_train'),double(tr_labels),'Coding','onevsall','Learners',tEnsemble,'Prior','uniform','Options',options);

%CVMdl = crossval(Mdl,'Options',options);




