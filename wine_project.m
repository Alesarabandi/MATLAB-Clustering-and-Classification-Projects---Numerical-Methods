%% WINE DATASET
clear
close all

% Load Wine dataset
load WineData.mat

% K-MEDOIDS ALGORITHM
[n,p] = size(X);

% Calculate distance matrix D using Inf norm
D = zeros(p);
for i = 1:p
    for j = i:p
        if (i == j)
            D(i,j) = 0;
        else
            D(i,j) = norm(X(:,i) - X(:,j), Inf);
        end
    end
end
D = D + D';

rng(407); % Set seed for replicability
k = 3;
I_m = sort(randperm(p, k)); % Pick k random indices as initial medoids

starting_medoids = I_m; % Initial medoids indices

figure()
plot(X(1,:), X(2,:), 'bo', 'MarkerSize', 8);
hold on;
plot(X(1,I_m), X(2,I_m), 'xr', 'MarkerSize', 8);
xlabel('x')
ylabel('y')
legend('Data', 'Initial medoids')

Err = 1;
itmax = 100;
tol = 1.0e-14;
iter = 0;
while(iter < itmax && Err > tol)
    % Assignment step
    D_m = D(:, I_m); % Distances w.r.t. medoids submatrix
    [~, I_assign] = min(D_m'); % Index to clusters
    Q = sum(min(D_m)); % Efficient strategy for PAM

    oldI_m = I_m;

    if (iter == 0)
        % Plot initial clusters
        figure()
        for j = 1:k
            X_l{j} = X(:, find(I_assign == j)); % j-th cluster
        end
        scatter3(X_l{1}(1,:), X_l{1}(2,:), X_l{1}(3,:), 'red')
        hold on
        scatter3(X_l{2}(1,:), X_l{2}(2,:), X_l{2}(3,:), 'blue')
        hold on
        scatter3(X_l{3}(1,:), X_l{3}(2,:), X_l{3}(3,:), 'green')
        title('Initial clustering')
    end

    % Updating step
    for ell = 1:k
        I_ell = find(I_assign == ell); % Indices to points in the cluster
        D_ell = D(I_ell, I_ell);
        [~, j] = min(sum(D_ell));

        % Update medoid indices
        I_m(ell) = I_ell(j);
    end

    % Recompute global coherence
    Qnew = sum(min(D(:, I_m)));

    % If not converged, continue from the assignment step
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

disp('flag')
flag

final_medoids = I_m;

% Q plot
figure()
semilogy([1:iter],Qplot,'bo-')
[[1:iter]' Qplot(:)];
xlabel('Iteration');
ylabel('Q value');

% Final Clusters
figure()
for j = 1:k
    X_l{j} = X(:, find(I_assign == j)); % j-th cluster
end


scatter3(X_l{1}(1,:), X_l{1}(2,:), X_l{1}(3,:), 'red')
hold on
scatter3(X_l{2}(1,:), X_l{2}(2,:), X_l{2}(3,:), 'blue')
hold on
scatter3(X_l{3}(1,:), X_l{3}(2,:), X_l{3}(3,:), 'green')
title('Final clustering')

figure()
scatter(X(1,:), X(2,:));
hold on;
scatter(X(1,I_m), X(2,I_m), 'xr');
xlabel('x')
ylabel('y')
legend('Data', 'Final medoids')

figure()
cm = confusionchart(I, I_assign);

% K-MEANS ALGORITHM
X = X'; % Transpose X for kmeans
rng(2012);
[idx, C] = kmeans(X, 3); % Perform k-means clustering

figure;
plot(X(idx == 1, 1), X(idx == 1, 2), 'r.', 'MarkerSize', 9)
hold on
plot(X(idx == 2, 1), X(idx == 2, 2), 'b.', 'MarkerSize', 9)
plot(X(idx == 3, 1), X(idx == 3, 2), 'g.', 'MarkerSize', 9)
plot(C(:, 1), C(:, 2), 'co', 'MarkerSize', 9, 'LineWidth', 1.5)
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids', 'Location', 'NW')
title('Cluster Assignments and Centroids')
hold off

x1 = min(X(:,1)):0.01:max(X(:,1));
x2 = min(X(:,2)):0.01:max(X(:,2));
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot

idx2Region = kmeans(XGrid, 3);

figure()
cm = confusionchart(I, idx);