disp("starting")
train = csvread('mnist_train_1500.csv');
trainsetlabels = train(:,785);
train = train(:,1:784);
train(:,785) = zeros(1500,1);
centroids = initialize_centroids(train,10);
for i=1:1500
    [index, vec_distance] = assign_vector_to_centroid(train(i,:),centroids);
    train(i,785) = index;
end


function y=initialize_centroids(data,num_centroids)
data=data(:,1:end-1);

random_index=randperm(size(data,1));

centroids=data(random_index(1:num_centroids),:);

y=centroids;

end


function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
  data=data(:,1:end-1);
  distances = zeros(size(centroids,1),1);
  for i=1:size(centroids,1)
    distances(i)=(norm(data-centroids(i,:)))^2; 
    % Making an array of distances (squared norm) between vectors and centroids
  end
  [vec_distance, index] = min(distances);
end

function new_centroids = update_Centroids(data,K)
  new_centroids = randi([0 255],K, size(data,2)-1); % initialize new centroids to random values
  for i=1:K
    cluster_data=data(data(:,end)==i,1:end-1);
    if ~isempty(cluster_data)
      new_centroids(i,:) = mean(cluster_data,1);
    end
  end  
end
disp("complete")