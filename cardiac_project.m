clear
close all
load CardiacSPECT.mat

[n, p] = size(X);

D = zeros(p);

for i = 1:(p-1)
    for j = (i+1):p
        D(i, j) = sum(X(:, i) ~= X(:, j)) / n;
    end
end

D = D + D';

rng(2000); % Set seed for replicability
k = 2;
for n_init = 1:20
    I_m{n_init} = sort(randperm(length(D), k)); % Pick k random indices
    D_m{n_init} = D(:, I_m{n_init}); % Distances w.r.t. medoids submatrix
    [q{n_init}, I_assign{n_init}] = min(D_m{n_init}'); % Index to clusters
    Q(n_init) = sum(q{n_init}); % Efficient strategy for PAM
end

[lowest_tightness, iteration_lowest_tightness] = min(Q');

I_m = I_m{iteration_lowest_tightness}; % Medoids with lowest overall coherence
starting_medoids = I_m; % Starting medoid indices

Err = 1;
itmax = 100;
tol = 1.0e-10;
iter = 0;
while(iter < itmax && Err > tol)
    % Assignment step
    D_m = D(1:end, I_m); % Distances w.r.t. medoids submatrix
    [q, I_assign] = min(D_m'); % Index to clusters
    Q = sum(q); % Efficient strategy for PAM
    oldI_m = I_m;

    % Updating step
    for ell = 1:k
        I_ell = find(I_assign == ell); % Indices to points in the cluster
        D_ell = D(I_ell, I_ell);
        [qq(ell), j] = min(sum(D_ell));

        % Update medoid indices
        I_m(ell) = I_ell(j);
    end

    % Recompute global coherence
    Qnew = sum(qq);

    % If not converged, continue from the assignment step
    Err = abs(Q - Qnew);
    Q = Qnew;
    Qplot(iter+1) = Q;
    iter = iter + 1;

    if (Err < tol)
        flag = 0;
    else
        flag = 1;
    end
end

disp('flag')
flag

% Final medoids
final_medoids = I_m;

% Q plot
figure()
semilogy([1:iter], Qplot, '-o ')
[[1:iter]' Qplot(:)];

figure()
cm = confusionchart(I, I_assign-1);

%% Create matrix C
% c11: Number of 1's in your cluster A
% c12: Number of 1's in your cluster B
% c21: Number of 0's in your cluster A
% c22: Number of 0's in your cluster B

% Compare I and I_assign
I_assign = I_assign - 1;

% Cluster A = Abnormal (1)
% Cluster B = Normal (0)

c11 = sum(I_assign == 1 & I == 1);
c12 = sum(I_assign == 1 & I == 0);
c21 = sum(I_assign == 0 & I == 1);
c22 = sum(I_assign == 0 & I == 0);

C = [c11 c12; c21 c22];

disp('Matrix C:');
disp(C);

%% Create matrix C
% c11: Number of 1's in your cluster B
% c12: Number of 1's in your cluster A
% c21: Number of 0's in your cluster B
% c22: Number of 0's in your cluster A

C_2 = [c12 c11; c22 c21];

disp('Matrix C_2:');
disp(C_2);
