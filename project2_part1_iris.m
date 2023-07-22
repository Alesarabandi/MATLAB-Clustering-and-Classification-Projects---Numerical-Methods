close all
clear

load('IrisDataAnnotated.mat')

X = X'; % Transpose the data matrix to match the expected format
rng(2012);

% Perform the first K-Means clustering
[idx, C] = kmeans(X, 3);

figure;
plot(X(idx==1, 1), X(idx==1, 2), 'r.', 'MarkerSize', 9)
hold on
plot(X(idx==2, 1), X(idx==2, 2), 'b.', 'MarkerSize', 9)
plot(X(idx==3, 1), X(idx==3, 2), 'g.', 'MarkerSize', 9)
plot(C(:, 1), C(:, 2), 'co', 'MarkerSize', 9, 'LineWidth', 1.5)
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids', 'Location', 'NW')
title('Cluster Assignments and Centroids')
hold off

x1 = min(X(:, 1)):0.01:max(X(:, 1));
x2 = min(X(:, 2)):0.01:max(X(:, 2));
[x1G, x2G] = meshgrid(x1, x2);

figure()
cm = confusionchart(I, idx);

clear
load('IrisDataAnnotated.mat')

% Perform the K-Medoids clustering
[n, p] = size(X);

rng(407);
% Calculate the pairwise distances using the Manhattan distance (L1 norm)
D = zeros(p, p);
for i = 1:p
    for j = i:p
        if i == j
            D(i, j) = 0;
        else
            D(i, j) = norm(X(:, i) - X(:, j), 1);
        end
    end
end
D = D + D';

rng(2012); % Set the random seed for replicability
k = 3; % Number of clusters
I_m = sort(randperm(p, k)); % Randomly initialize the medoids' indices

starting_medoids = I_m; % Starting medoids indices

figure()
plot(X(1, :), X(2, :), 'bo', 'MarkerSize', 8);
hold on;
plot(X(1, I_m), X(2, I_m), 'xr', 'MarkerSize', 8);
xlabel('x')
ylabel('y')
legend('Data', 'Initial medoids')

Err = 1;
itmax = 100;
tol = 1.0e-10;
iter = 0;
while (iter < itmax && Err > tol)
    % 1. Assignment step

    D_m = D(:, I_m); % Distances with respect to medoids submatrix

    [~, I_assign] = min(D_m, [], 2); % Indices of assigned clusters

    Q = sum(min(D_m, [], 2)); % Efficient strategy for PAM

    oldI_m = I_m;

    if (iter == 0)
        % Plot initial clusters
        figure()
        for j = 1:k
            X_l{j} = X(:, I_assign == j); % Points in the j-th cluster
        end
        scatter3(X_l{1}(1, :), X_l{1}(2, :), X_l{1}(3, :), 'red')
        hold on
        scatter3(X_l{2}(1, :), X_l{2}(2, :), X_l{2}(3, :), 'blue')
        hold on
        scatter3(X_l{3}(1, :), X_l{3}(2, :), X_l{3}(3, :), 'green')
        title('Initial clustering')
    end

    % 2. Updating step
    for ell = 1:k
        I_ell = find(I_assign == ell); % Indices of points in the cluster

        D_ell = D(I_ell, I_ell); % Submatrix of distances

        [~, j] = min(sum(D_ell)); % Find the medoid index within the cluster

        I_m(ell) = I_ell(j); % Update the medoid index
    end

    % Recompute the global coherence
    Qnew = sum(min(D(:, I_m), [], 2));

    % Check convergence
    Err = abs(Q - Qnew);
    Q = Qnew;
    Qplot(iter + 1) = Q;

    iter = iter + 1;

    if (Err < tol)
        flag = 0;
    else
        flag = 1;
    end
end

disp('flag') % To see which criteria we used
flag

final_medoids = I_m; % Final medoid indices

% Q plot
figure()
semilogy(1:iter, Qplot, 'bo-')
[[1:iter]', Qplot(:)];

% Final Clusters
figure()
for j = 1:k
    X_l{j} = X(:, I_assign == j); % Points in the j-th cluster
end

scatter3(X_l{1}(1, :), X_l{1}(2, :), X_l{1}(3, :), 'red')
hold on
scatter3(X_l{2}(1, :), X_l{2}(2, :), X_l{2}(3, :), 'blue')
hold on
scatter3(X_l{3}(1, :), X_l{3}(2, :), X_l{3}(3, :), 'green')
title('Final clustering')

figure()
scatter(X(1, :), X(2, :));
hold on;
scatter(X(1, I_m), X(2, I_m), 'xr');
xlabel('x')
ylabel('y')
legend('Data', 'Final medoids')

figure()
cm = confusionchart(I, I_assign);

