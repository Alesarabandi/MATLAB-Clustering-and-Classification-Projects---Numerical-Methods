% Load the data from the file
load('CongressionalVoteData.mat')

% Remove columns with all zeros (missing votes)
col_nums = find(~any(X,1));
X(:,col_nums) = [];
I(:,col_nums) = [];

% Compute the dissimilarity matrix
D = zeros(size(X,2),size(X,2));
for i = 1:size(X,2)
    for j = i:size(X,2)
        vector = zeros(size(X,1),1);
        for k = 1:size(X,1)
            if (X(k,i) == X(k,j))
                vector(k) = 0;
            else
                vector(k) = 1;
            end
        end
        D(i,j) = (sum(vector))/size(X,1);
    end
end
D = D+D';

% Set initial medoids
medoids=[50,100];

% Plot the data with initial medoids
figure()
plot(X(1,:),X(2,:),'bo','MarkerSize',5);
hold on ;
plot(X(1,medoids),X(2,medoids),"xr",'MarkerSize',8);
xlim([-2,2]);
ylim([-2,2]);
xlabel('x')
ylabel('y')
title('Congress Dataset clustering with initial medoids')

% Set up variables for the k-medoids algorithm
Error=1;
max_iter = 1000;
tol=1.0e-10;
iter=0;

% Run the k-medoids algorithm
while(iter < max_iter && Error >= tol)
    distance_matrix = D(1:end,medoids);
    [q,index] = min(distance_matrix');
    Q = sum(q);
    k = length(medoids);
    old_medoids=medoids;
    clear medoids
    for l=1:k
        I_l = find(index == l);
        distance_l = D(I_l,I_l);
        s = sum(distance_l);
        [qq(l),j] = min(sum(distance_l));
        medoids(l) = I_l(j);
        new_medoids = medoids;
    end
    Q_new=sum(qq);
    Error = abs(Q-Q_new);
    Q=Q_new;
    iter = iter+1;
    if (Error < tol)
        flag=0;
    else
        flag=1;
    end
end

% Store final medoids
final_medoids = medoids;

% Plot the data with final medoids
figure()
plot(X(1,:),X(2,:),'bo','MarkerSize',5);
hold on ;
plot(X(1,medoids),X(2,medoids),"xr");
xlabel('x')
ylabel('y')
xlim([-2,2]);
ylim([-2,2]);
title('Congress Dataset with k-medoids clustering')

% Generate confusion matrix
figure()
cm = confusionchart(sort(I),sort(index-1))
C = cm.NormalizedValues
