clear
close all

load BiopsyData.mat X

X = rmmissing(X');
X = X';

[n,p] = size(X);

% Compute the pairwise distances using the Inf norm
D = zeros(p);
for i = 1:p
    if i == p
        D(i, p) = 0;
    else
        for j = i+1:p
            D(i, j) = norm(X(:, i) - X(:, j), Inf); % or D(i, j) = sum(abs(X(:, i) - X(:, j)))
            % TRY ALSO NORM 1 and NORM 2
            % compare also the iteration number
        end
    end
end

D = D + D';

Err = 1;
itmax = 100;
tau = 1.0e-10;
k = 2;
iter = 0;

[n, p] = size(D);
for n_init = 1:20
    I_m{n_init} = sort(randperm(length(D), k)); % pick up k random indices

    D_m{n_init} = D(:, I_m{n_init}); % distances w.r.t. medoids submatrix

    [q{n_init}, I_assign{n_init}] = min(D_m{n_init}'); % index to clusters

    Q(n_init) = sum(q{n_init}); % efficient strategy for PAM
end

[lowest_tightness, iteration_lowest_tightness] = min(Q);

I_m = I_m{iteration_lowest_tightness}; % choose the medoids with lowest overall coherence

starting_medoids = I_m;

%% Plot of the initial medoids
figure()
plot(X(1, :), X(2, :), 'bo', 'MarkerSize', 8);
hold on;
plot(X(1, I_m), X(2, I_m), 'xr', 'MarkerSize', 8);
xlabel("x")
ylabel("y")
legend('Data', 'Initial medoids')

Err = 1;
itmax = 100;
tau = 1.0e-10;
iter = 0;

clear q
clear Q

while iter < itmax && Err > tau
    D_m = D(:, I_m); % distances w.r.t. medoids submatrix

    [q, I_assign] = min(D_m'); % index to clusters

    Q(iter + 1) = sum(q);

    if iter == 0
        %% Plot initial clusters
        figure()
        for j = 1:k
            X_l{j} = X(:, I_assign == j); % k-th cluster
        end
        scatter3(X_l{1}(1, :), X_l{1}(2, :), X_l{1}(3, :), 'red')
        hold on
        scatter3(X_l{2}(1, :), X_l{2}(2, :), X_l{2}(3, :), 'blue')
        title('Initial clustering')
    end

    %% Updating step
    for ell = 1:k
        I_ell = find(I_assign == ell); % Indices to points in the cluster

        D_ell = D(I_ell, I_ell);

        [qq(ell), j] = min(sum(D_ell)); % Within-cluster coherence of each cluster

        I_m(ell) = I_ell(j); % Updating medoid indexes
    end

    %% Recompute the global coherence
    Q(iter + 2) = sum(qq);
    %% If not converged, continue from the assignment step
    Err = abs(Q(iter + 1) - Q(iter + 2));

    Errplot(iter + 1) = Err;

    iter = iter + 1;

    if Err < tau
        flag = 0;
    else
        flag = 1;
    end
end
I_assign = I_assign;
I_bar = I_m;

%% Q plot
figure()
semilogy([1:iter+1], Q, 'bo-');
[[1:iter+1]' Q(:)];

%% Error plot
figure()
semilogy([1:iter], Errplot, 'bo-');
[[1:iter]' Errplot(:)];

%% Final medoids
figure()
scatter(X(1, :), X(2, :));
hold on;
scatter(X(1, I_m), X(2, I_m), 'xr');
xlabel("x")
ylabel("y")
legend('Data', 'Final medoids')

%% Final Clusters
figure()
for j = 1:k
    X_l{j} = X(:, I_assign == j); % k-th cluster
end
scatter3(X_l{1}(1, :), X_l{1}(2, :), X_l{1}(3, :), 'red')
hold on
scatter3(X_l{2}(1, :), X_l{2}(2, :), X_l{2}(3, :), 'blue')
title('Final clustering')
