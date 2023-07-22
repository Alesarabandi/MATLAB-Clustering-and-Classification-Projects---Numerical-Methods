clear
close all

load CongressionalVoteData.mat

delete = find(all(X == 0)); % Find columns with all zeros

X(:, delete) = [];
I(:, delete) = [];

[n, p] = size(X);

D = zeros(p); % Initialize the distance matrix with zeros

% Compute dissimilarity matrix
for i = 1:p-1
    for j = i+1:p
        invalid = sum(any([X(:, i) X(:, j)]' == 0)); % Count number of invalid entries (zeros)
        double0 = sum(all([X(:, i) X(:, j)]' == 0)); % Count number of entries with both zeros
        if invalid == 16
            D(i, j) = 0.5; % If the pair has both zeros, set the dissimilarity to 0.5
        else
            D(i, j) = (sum(X(:, i) ~= X(:, j)) - invalid + double0) / (n - invalid); % Calculate dissimilarity
        end
    end
end

D = D + D'; % Symmetricize the distance matrix

rng(500); % Set seed for replicability
k = 2;

% Apply k-medoids algorithm
for n_init = 1:20
    I_m{n_init} = sort(randperm(length(D), k)); % Select initial medoids randomly
    D_m{n_init} = D(:, I_m{n_init});
    [q{n_init}, I_assign{n_init}] = min(D_m{n_init}');
    Q(n_init) = sum(q{n_init}); % Compute total dissimilarity (objective function)
end

[lowest_tightness, iteration_lowest_tightness] = min(Q');
I_m = I_m{iteration_lowest_tightness}; % Select the medoids with the lowest overall dissimilarity
starting_medoids = I_m;

Err = 1;
itmax = 100;
tol = 1.0e-10;
iter = 0;

% K-medoids iterations
while iter < itmax && Err > tol
    % Assignment step
    D_m = D(:, I_m); % Distances w.r.t. medoids submatrix
    [q, I_assign] = min(D_m'); % Index to clusters
    Q = sum(q); % Total dissimilarity

    % Updating step
    for ell = 1:k
        I_ell = find(I_assign == ell); % Indices to points in the cluster
        D_ell = D(I_ell, I_ell);
        [qq(ell), j] = min(sum(D_ell));
        I_m(ell) = I_ell(j); % Swap (updating) of the medoids indices
    end

    Qnew = sum(qq); % Compute new total dissimilarity

    % Check convergence
    Err = abs(Q - Qnew);
    Q = Qnew;
    Qplot(iter + 1) = Q;

    iter = iter + 1;

    if Err < tol
        flag = 0;
    else
        flag = 1;
    end
end
disp('flag')
flag

% Final medoids
final_medoids = I_m

% Plot Q
figure()
semilogy([1:iter], Qplot, '-o')
[[1:iter]' Qplot(:)];

% Confusion matrix
figure()
cm = confusionchart(I, I_assign - 1)

% Create matrix C
c11 = sum(I_assign == 1 & I == 1);
c12 = sum(I_assign == 1 & I == 0);
c21 = sum(I_assign == 0 & I == 1);
c22 = sum(I_assign == 0 & I == 0);

C = [c11 c12; c21 c22];

% Display matrix C
disp('Matrix C:');
disp(C);

% Create alternative matrix C_2
C_2 = [c12 c11; c22 c21];

% Display alternative matrix C_2
disp('Alternative Matrix C_2:');
disp(C_2);