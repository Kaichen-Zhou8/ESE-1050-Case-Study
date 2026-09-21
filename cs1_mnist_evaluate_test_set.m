%% This code evaluates the test set.

% ** Important.  This script requires that:
% 1)'centroid_labels' be established in the workspace
% AND
% 2)'centroids' be established in the workspace
% AND
% 3)'test' be established in the workspace
% testing set (200 images with 11 outliers)
test=csvread('mnist_test_200.csv');
% store the correct test labels
correctlabels = test(:,785);
test=test(:,1:784);


% IMPORTANT!!:
% You should save 1) and 2) in a file named 'classifierdata.mat' as part of
% your submission.
load ('classifierdata.mat');
predictions = zeros(200,1);
outliers = zeros(200,1);
distances = zeros(200,1);
% loop through the test set, figure out the predicted number
for i = 1:200

testing_vector=test(i,:);

% Extract the centroid that is closest to the test image
[prediction_index, vec_distance]=assign_vector_to_centroid(testing_vector,centroids);

predictions(i) = centroid_labels(prediction_index);
distances(i) = vec_distance;

end
disp("Average distance to closest centroid: " + mean(distances));

%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0


%% MAKE A STEM PLOT OF THE OUTLIER FLAG
figure;
% FILL IN

%% The following plots the correct and incorrect predictions
% Make sure you understand how this plot is constructed
figure;
plot(correctlabels,'o');
hold on;
plot(predictions,'x');
title('Predictions');

%% The following line provides the number of instances where and entry in correctlabel is
% equatl to the corresponding entry in prediction
% However, remember that some of these are outliers
sum(correctlabels==predictions)

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
  distances = zeros(size(centroids,1),1);
  for i=1:size(centroids,1)
    distances(i)=(norm(data-centroids(i,:)))^2; 
    % Making an array of distances (squared norm) between vectors and centroids
  end
  [vec_distance, index] = min(distances);
end

