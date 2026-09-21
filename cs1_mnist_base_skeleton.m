
clear all;
close all;
rng(1);

%% In this script, you need to implement three functions as part of the k-means algorithm.
% These steps will be repeated until the algorithm converges:

  % 1. initialize_centroids
  % This function sets the initial values of the centroids

  % 2. assign_vector_to_centroid
  % This goes through the collection of all vectors and assigns them to
  % centroid based on norm/distance

  % 3. update_centroids
  % This function updates the location of the centroids based on the collection
  % of vectors (handwritten digits) that have been assigned to that centroid.


%% Initialize Data Set
% These next lines of code read in two sets of MNIST digits that will be used for training and testing respectively.

% training set (1500 images)
train = csvread('mnist_train_1500.csv');
trainsetlabels = train(:,785);
train = train(:,1:784);
train(:,785) = zeros(1500,1);

% testing set (200 images with 11 outliers)
test=csvread('mnist_test_200.csv');
% store the correct test labels
correctlabels = test(:,785);
test=test(:,1:784);

% now, zero out the labels in "test" so that you can use this to assign
% your own predictions and evaluate against "correctlabels"
% in the 'cs1_mnist_evaluate_test_set.m' script
test(:,785)=zeros(200,1);

%% After initializing, you will have the following variables in your workspace:
% 1. train (a 1500 x 785 array, containins the 1500 training images)
% 2. test (a 200 x 785 array, containing the 200 testing images)
% 3. correctlabels (a 200 x 1 array containing the correct labels (numerical
% meaning) of the 200 test images

%% To visualize an image, you need to reshape it from a 784 dimensional array into a 28 x 28 array.
% to do this, you need to use the reshape command, along with the transpose
% operation.  For example, the following lines plot the first test image

figure;
colormap('gray'); % this tells MATLAB to depict the image in grayscale
testimage = reshape(test(1,[1:784]), [28 28]);
% we are reshaping the first row of 'test', columns 1-784 (since the 785th
% column is going to be used for storing the centroid assignment.
imagesc(testimage'); % this command plots an array as an image.  Type 'help imagesc' to learn more.

%% After importing, the array 'train' consists of 1500 rows and 785 columns.
% Each row corresponds to a different handwritten digit (28 x 28 = 784)
% plus the last column, which is used to index that row (i.e., label which
% cluster it belongs to.  Initially, this last column is set to all zeros,
% since there are no clusters yet established.

%% This next section of code calls the three functions you are asked to specify
% k=30 was chosen from the sweep in k_sweep.m, accuracy levels off past this
% point so the extra centroids buy very little
k= 30; % set k
max_iter= 20; % set the number of iterations of the algorithm
% k-means only finds a local optimum, so we try several random starts
num_restarts = 5;

%% This section runs k-means num_restarts times and keeps the best start
% each start converges somewhere different, we keep the lowest cost one

best_cost = Inf;
best_centroids = [];
best_assignments = [];
best_cost_iteration = zeros(max_iter, 1);

for restart = 1:num_restarts

    % a fresh random start for this attempt
    centroids = initialize_centroids(train,k);
    cost_iteration = zeros(max_iter, 1);

    %% This for-loop enacts the k-means algorithm
    for iter=1:max_iter
      % reset cost counter for this iteration
      total_cost = 0;
      for i=1:1500
          [index, vec_distance] = assign_vector_to_centroid(train(i,:),centroids);
          train(i,785) = index;
          % accumulate squared distance for this point
          total_cost = total_cost + vec_distance;
      end
      % save this iteration's total cost
      cost_iteration(iter) = total_cost;
      centroids=update_centroids(train,k);
    end

    % reassign once more so the cost matches the final centroids
    final_cost = 0;
    for i=1:1500
        [index, vec_distance] = assign_vector_to_centroid(train(i,:),centroids);
        train(i,785) = index;
        final_cost = final_cost + vec_distance;
    end

    % keep this start only if it beat the earlier ones
    if final_cost < best_cost
        best_cost = final_cost;
        best_centroids = centroids;
        best_assignments = train(:,785);
        best_cost_iteration = cost_iteration;
    end

    fprintf('restart %d/%d  cost = %.4e\n', restart, num_restarts, final_cost);
end

% everything below uses the winning start
centroids = best_centroids;
train(:,785) = best_assignments;
cost_iteration = best_cost_iteration;
fprintf('best cost = %.4e\n', best_cost);


%% This section of code plots the k-means cost as a function of the number
% of iterations

figure;
plot(1:max_iter, cost_iteration, '-o');
xlabel('Iteration');
ylabel('K-means Cost (Sum of Squared Distances)');
title('K-means Cost vs. Iteration');
grid on;
set(gca,'YScale','log');


%% This next section of code will make a plot of all of the centroids
% Again, use help <functionname> to learn about the different functions
% that are being used here.

figure;
colormap('gray');

plotsize = ceil(sqrt(k));

for ind=1:k

    centr=centroids(ind,[1:784]);
    subplot(plotsize,plotsize,ind);

    imagesc(reshape(centr,[28 28])');
    title(strcat('Centroid ',num2str(ind)))

end

centroid_labels = zeros(k,1);
for i=1:k
    cluster_data=train(train(:,785)==i,1:end-1);
    if ~isempty(cluster_data)
        centroid_labels(i) = mode(trainsetlabels(train(:,785)==i));
    else
        centroid_labels(i) = -1; % indicate no label assigned
    end
end
disp('Centroid Labels:');
for i=1:k
    fprintf('Centroid %d: Label %d\n', i, centroid_labels(i));
end

%% Save the classifier data for the competition phase
% pad to 785 columns so the array matches the required k x 785 format
centroids(:,785) = 0;
% use doubles, uint8 would clamp the -1 flag to 0 and mislabel that centroid
centroid_labels = double(centroid_labels);
save('classifierdata.mat','centroids','centroid_labels');
disp('classifierdata.mat saved.');


%% Function to initialize the centroids
% This function randomly chooses k vectors from our training set and uses them to be our initial centroids
% There are other ways you might initialize centroids.
% ***Feel free to experiment.***
% Note that this function takes two inputs and emits one output (y).


function y=initialize_centroids(data,num_centroids)
data=data(:,1:end-1);

random_index=randperm(size(data,1));

centroids=data(random_index(1:num_centroids),:);

y=centroids;

end
% The input of the function is as follows:
% Input 1: it should basically always be the training set (train), no need to for any indexing or modification of the training set.
% Input 2: the number of centroids (k)

%% Function to pick the Closest Centroid using norm/distance
% This function takes two arguments, a vector and a set of centroids
% It returns the index of the assigned centroid and the distance between
% the vector and the assigned centroid.

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
  % use columns 1-784 only, column 785 stores the cluster assignment
  data = data(1,1:784);
  distances = zeros(size(centroids,1),1);
  for i = 1:size(centroids,1)
    distances(i) = norm(data - centroids(i,1:784))^2;
    % Making an array of distances (squared norm) between vectors and centroids
  end
  [vec_distance, index] = min(distances);
end

%The input of the function is as follows:
% Input 1: a vector (a row of the training set): train(i,:)
% Input 2: the set of centroids (centroids)

%% Function to compute new centroids using the mean of the vectors currently assigned to the centroid.
% This function takes the set of training images and the value of k.
% It returns a new set of centroids based on the current assignment of the
% training images.

function new_centroids = update_centroids(data,K)
  % preallocate so the array does not grow inside the loop
  new_centroids = zeros(K, size(data,2)-1);
  for i=1:K
    cluster_data=data(data(:,end)==i,1:end-1);
    if ~isempty(cluster_data)
      % centroid is the average of its assigned images
      new_centroids(i,:) = mean(cluster_data,1);
    else
      random_row = randi(size(data,1));
      % if cluster is empty, reset to a random training image
      new_centroids(i,:) = data(random_row,1:end-1);
    end
  end
end
