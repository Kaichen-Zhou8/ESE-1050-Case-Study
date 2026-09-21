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
<<<<<<< HEAD
outliers = zeros(200,1);
distances = zeros(200,1);
k=size(centroids,1);
=======


>>>>>>> 7447954 (Add multi-restart k-means, confusion analysis, and k sweep)
% loop through the test set, figure out the predicted number
for i = 1:200

testing_vector=test(i,:);

% Extract the centroid that is closest to the test image
[prediction_index, vec_distance]=assign_vector_to_centroid(testing_vector,centroids);

predictions(i) = centroid_labels(prediction_index);
distances(i) = vec_distance;

end

%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
<<<<<<< HEAD
for i = 1:200
    if distances(i) > 4.1*10^6
        outliers(i) = 1;
    end
end

=======
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0
% flag samples with pixel values outside the valid MNIST range [0, 255]
outliers = any(test(:,1:784) < 0 | test(:,1:784) > 255, 2);
>>>>>>> 7447954 (Add multi-restart k-means, confusion analysis, and k sweep)

%% MAKE A STEM PLOT OF THE OUTLIER FLAG
% plot the detected outlier flags
figure;
<<<<<<< HEAD
stem(outliers);
figure
colormap('gray');
plotsize = ceil(sqrt(k));
plot_ind=1;
for ind=1:200    
    if outliers(ind) == 1
        outlier_image=test(ind,:);
        subplot(plotsize,plotsize,plot_ind);
        imagesc(reshape(outlier_image,[28 28])');
        title(strcat('Outlier at:',num2str(ind)))
        plot_ind = plot_ind + 1;
    end
end
=======
stem(1:200, outliers, 'filled');
xlabel('Test image index');
ylabel('Outlier flag');
title('Outlier Detection');
ylim([-0.1 1.1]);
grid on;
>>>>>>> 7447954 (Add multi-restart k-means, confusion analysis, and k sweep)

%% The following plots the correct and incorrect predictions
% Make sure you understand how this plot is constructed
% compare the correct labels with the predicted labels
figure;
plot(correctlabels,'o','LineWidth',1.5,'MarkerSize',8);
hold on;
plot(predictions,'x','LineWidth',1.5,'MarkerSize',8);
xlabel('Test Set Index');
ylabel('Label');
title('Predictions');
legend('Correct Labels','Predicted Labels','Location','best');
grid on;

%% The following line provides the number of instances where and entry in correctlabel is
% equal to the corresponding entry in prediction
% However, remember that some of these are outliers
disp("Correct predictions out of 200: " + sum(correctlabels==predictions));
disp("Number of outliers: " + sum(outliers));
disp("Average distance to closest centroid: " + mean(distances));
figure
bar (distances);
yscale('log');

<<<<<<< HEAD
function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
  distances = zeros(size(centroids,1),1);
  for i=1:size(centroids,1)
    distances(i)=(norm(data-centroids(i,:)))^2; 
    % Making an array of distances (squared norm) between vectors and centroids
  end
  [vec_distance, index] = min(distances);
=======
% print the accuracy/outlier outputs
normal_correct = sum(correctlabels(~outliers) == predictions(~outliers));
normal_total = sum(~outliers);
normal_accuracy = normal_correct / normal_total;

fprintf('Detected outliers: %d\n', sum(outliers));
fprintf('Normal samples correct: %d/%d\n', normal_correct, normal_total);
fprintf('Non-outlier accuracy: %.4f\n', normal_accuracy);

%% Build a confusion matrix over the non-outlier images
% rows are the true digit, columns are what the classifier predicted
% digits run 0-9 so everything is shifted by one to be used as an index
confusion = zeros(10,10);
for i = 1:200
    if ~outliers(i)
        r = correctlabels(i) + 1;
        c = predictions(i) + 1;
        confusion(r,c) = confusion(r,c) + 1;
    end
>>>>>>> 7447954 (Add multi-restart k-means, confusion analysis, and k sweep)
end

disp(' ');
disp('Confusion matrix, rows = true digit 0-9, columns = predicted 0-9:');
disp(confusion);

%% Report accuracy for each digit separately
% this shows which digits the classifier handles worst
fprintf('Per digit accuracy:\n');
for d = 0:9
    is_d = (correctlabels==d) & ~outliers;
    if any(is_d)
        fprintf('  digit %d : %2d/%2d correct\n', d, sum(predictions(is_d)==d), sum(is_d));
    end
end

%% List the confusions that happened most often
% copy the matrix and clear the diagonal so only the mistakes are left
mistakes = confusion;
for d = 1:10
    mistakes(d,d) = 0;
end

fprintf('\nMost common confusions:\n');
for n = 1:5
    [count, pos] = max(mistakes(:));
    if count == 0
        break
    end
    [true_digit, pred_digit] = ind2sub(size(mistakes), pos);
    fprintf('  true %d predicted as %d : %d times\n', true_digit-1, pred_digit-1, count);
    % clear it so the next loop finds the next largest
    mistakes(true_digit,pred_digit) = 0;
end

%% Show the non-outlier images that were classified wrong
% these are the images to comment on for the bonus question
wrong = find(predictions ~= correctlabels & ~outliers);

figure;
colormap('gray');
plotsize = ceil(sqrt(length(wrong)));
for n = 1:length(wrong)
    i = wrong(n);
    subplot(plotsize,plotsize,n);
    imagesc(reshape(test(i,1:784),[28 28])');
    title(sprintf('%d as %d', correctlabels(i), predictions(i)));
    axis off;
end
sgtitle('Misclassified test images (true as predicted)');

%% Show the 11 outlier images on their own
% worth a look since they are corrupted in a specific way
out_idx = find(outliers);

figure;
colormap('gray');
for n = 1:length(out_idx)
    i = out_idx(n);
    subplot(3,4,n);
    imagesc(reshape(test(i,1:784),[28 28])');
    title(sprintf('%d as %d', correctlabels(i), predictions(i)));
    axis off;
end
sgtitle('Outlier test images (true as predicted)');

% the largest pixel value shows how badly each outlier is scaled
fprintf('\nOutlier images:\n');
for n = 1:length(out_idx)
    i = out_idx(n);
    fprintf('  index %3d : true %d, predicted %d, max pixel %6.1f\n', ...
            i, correctlabels(i), predictions(i), max(test(i,1:784)));
end
fprintf('Outliers classified correctly: %d/%d\n', ...
        sum(correctlabels(outliers)==predictions(outliers)), sum(outliers));

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
