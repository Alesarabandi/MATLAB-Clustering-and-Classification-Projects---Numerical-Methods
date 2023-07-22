function [I_assign, I_bar] = my_k_medoids(k, D, tau)
    % Initialization
    for init = 1:20
        I_m{init} = sort(randperm(length(D), k));  % Pick k random indices
        D_m{init} = D(:, I_m{init}); % Medoids submatrix
        [q{init}, I_assign{init}] = min(D_m{init}'); % Assign points to clusters
        Q(init) = sum(q{init}); % Compute overall coherence
    end

    % Find the iteration with the lowest tightness
    [lowest_tightness, iteration_lowest_tightness] = min(Q);

    I_m = I_m{iteration_lowest_tightness}; % Select medoids for lowest tightness

    starting_medoids = I_m;

    Err = 1;
    itmax = 100;
    iter = 0;

    % Assignment step
    while (iter < itmax && Err > tau)
        D_m = D(:, I_m); % Distances with respect to medoids submatrix

        [q, I_assign] = min(D_m'); % Assign points to clusters

        Q(iter + 1) = sum(q); % Compute overall coherence

        % Updating step
        for ell = 1:k
            I_ell = find(I_assign == ell); % Indices of points in the cluster

            D_ell = D(I_ell, I_ell); % Submatrix of distances

            [qq(ell), j] = min(sum(D_ell));
            I_m(ell) = I_ell(j); % Update the medoid index
        end

        % Recompute the global coherence
        Q(iter + 2) = sum(qq);

        % Check convergence
        Err = abs(Q(iter + 1) - Q(iter + 2));

        iter = iter + 1;
    end

    % Output
    I_assign = I_assign;
    I_bar = I_m;
end
