
clear all;
close all;
rng(1);

%% Compare several k values, with multiple restarts for each
% k-means depends on where the centroids start, so a single run can be bad
% for each k we try several starts and keep the one with the lowest cost

%% Initialize Data Set
% same data setup as the base skeleton

% training set (1500 images)
train = csvread('mnist_train_1500.csv');
trainsetlabels = train(:,785);
train = train(:,1:784);
train(:,785) = zeros(1500,1);

% testing set (200 images with 11 outliers)
test = csvread('mnist_test_200.csv');
correctlabels = test(:,785);
test = test(:,1:784);
test(:,785) = zeros(200,1);

% flag samples with pixel values outside the valid MNIST range [0, 255]
outliers = any(test(:,1:784) < 0 | test(:,1:784) > 255, 2);

%% Sweep settings
% add or remove k values here, nothing below needs editing
k_values = [10 20 30 40 50 60 70 80 90 100];
% how many random starts to try for each k
num_restarts = 5;
max_iter = 20;

accuracies = zeros(length(k_values),1);
best_costs = zeros(length(k_values),1);
% keep the best model for each k so it can be saved later
best_centroids_all = cell(length(k_values),1);
best_labels_all = cell(length(k_values),1);

%% Main sweep
for kk = 1:length(k_values)
    k = k_values(kk);

    % track the best start for this value of k
    best_cost = Inf;
    best_centroids = [];
    best_assignments = [];

    for restart = 1:num_restarts

        % run k-means from a fresh random start
        centroids = initialize_centroids(train,k);
        for iter = 1:max_iter
            for i = 1:1500
                [index, vec_distance] = assign_vector_to_centroid(train(i,:),centroids);
                train(i,785) = index;
            end
            centroids = update_centroids(train,k);
        end

        % reassign once more so the cost matches the final centroids
        final_cost = 0;
        for i = 1:1500
            [index, vec_distance] = assign_vector_to_centroid(train(i,:),centroids);
            train(i,785) = index;
            final_cost = final_cost + vec_distance;
        end

        % keep this start only if it beat the earlier ones
        if final_cost < best_cost
            best_cost = final_cost;
            best_centroids = centroids;
            best_assignments = train(:,785);
        end

        fprintf('  k = %2d  restart %d/%d  cost = %.4e\n', k, restart, num_restarts, final_cost);
    end

    % give each centroid the most common label among its assigned images
    centroid_labels = zeros(k,1);
    for i = 1:k
        if any(best_assignments==i)
            centroid_labels(i) = mode(trainsetlabels(best_assignments==i));
        else
            centroid_labels(i) = -1; % indicate no label assigned
        end
    end

    % classify the test set with the best centroids for this k
    predictions = zeros(200,1);
    for i = 1:200
        [prediction_index, vec_distance] = assign_vector_to_centroid(test(i,:),best_centroids);
        predictions(i) = centroid_labels(prediction_index);
    end

    % score on the non-outlier images only
    accuracies(kk) = sum(correctlabels(~outliers)==predictions(~outliers)) / sum(~outliers);
    best_costs(kk) = best_cost;
    best_centroids_all{kk} = best_centroids;
    best_labels_all{kk} = centroid_labels;

    fprintf('k = %2d : best cost = %.4e, accuracy = %.4f\n\n', k, best_cost, accuracies(kk));
end

%% Print the summary table for the report
disp('   k      best cost      accuracy');
for kk = 1:length(k_values)
    fprintf('%4d   %.4e   %.4f\n', k_values(kk), best_costs(kk), accuracies(kk));
end

%% Plot accuracy as a function of k
figure;
plot(k_values, accuracies, '-o','LineWidth',1.5,'MarkerSize',8);
xlabel('Number of clusters, k');
ylabel('Accuracy (non-outlier test images)');
title('Accuracy vs. k');
grid on;

%% Save the model for the chosen k
% set this to the k you picked from the plot above
k_for_submission = 20;
idx = find(k_values == k_for_submission);
centroids = best_centroids_all{idx};
centroid_labels = best_labels_all{idx};
% pad to 785 columns so the array matches the required k x 785 format
centroids(:,785) = 0;
save('classifierdata.mat','centroids','centroid_labels');
fprintf('classifierdata.mat saved using k = %d\n', k_for_submission);


%% Function to initialize the centroids
% This function randomly chooses k vectors from our training set and uses them to be our initial centroids

function y=initialize_centroids(data,num_centroids)
data=data(:,1:end-1);

random_index=randperm(size(data,1));

centroids=data(random_index(1:num_centroids),:);

y=centroids;

end

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
