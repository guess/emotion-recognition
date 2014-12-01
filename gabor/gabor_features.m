function [ gabor_features ] = gabor_features( data )
%GABOR_FATURES: Get the gabor features for N images of size D x D.
%   data:   D x D x N matrix

nfeat = 2560;   % number of gabor features
ga = gaborFilterBank(5, 8, 39, 39);

% Extract Gabor features from the data
data_size = size(data, 3);
gabor_features = zeros(nfeat, data_size);
for i=1:data_size
    gabor_features(:,i) = gaborFeatures(data(:,:,i), ga, 4, 4);
end

end

