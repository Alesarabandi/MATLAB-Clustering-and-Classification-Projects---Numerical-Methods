clear
close all

% Load the Wine dataset
load WineData.mat

% Split the dataset into three subsets based on class labels
I1 = find(I == 1);
I2 = find(I == 2);
I3 = find(I == 3);

X1 = X(:,I1);
X2 = X(:,I2);
X3 = X(:,I3);

% Calculate the class means and center the data within each class
p1 = length(I1);
c1 = 1/p1*sum(X1,2);
X1c = X1 - c1*ones(1,p1);

p2 = length(I2);
c2 = 1/p2*sum(X2,2);
X2c = X2 - c2*ones(1,p2);

p3 = length(I3);
c3 = 1/p3*sum(X3,2);
X3c = X3 - c3*ones(1,p3);

% Calculate the within-class scatter matrix (Sw) and between-class scatter matrix (Sb)
Sw = X1c*X1c' + X2c*X2c' + X3c*X3c';
p = p1 + p2 + p3;
c = 1/p * sum(X,2);
Sb = p1*(c1-c)*(c1-c)' + p2*(c2-c)*(c2-c)' + p3*(c3-c)*(c3-c)';

% Calculate the eigenvalues and eigenvectors of Sw
Xw = [X1c,X2c,X3c];
[eigenvectors, eigenvalues] = eig(Sw);

% Sort the eigenvalues in decreasing order
[eigenvalues, idx] = sort(diag(eigenvalues), 'descend');
eigenvectors = eigenvectors(:, idx);
% Check if the determinant of Sw is close to zero and apply regularization
% if necessary 
disp('det(Sw) before regularization')
det(Sw)
if (det(Sw)<=10^(-15))
    disp('do regularization')
    tau=10^(-10);
    epsilon=tau*max(eig(Sw));
    figure()
    semilogy(sXw2, '*')
    hold on
    semilogy(sXw2+epsilon,'sm')
    legend('eigen values in decreasing order', 'selected eigenvalues')
    sSw=eig(Sw);
    Sw=Sw+eye(size(Sw))*epsilon;
    sSwreg=eig(Sw);
    figure()
    semilogy(sSw,'r*')
    hold on 
    semilogy(sSwreg,'bd')
    legend('eig(Sw)', "eig(Sw reg)")

    disp('det (Sw) after regularization')
    det(Sw)
end

% Perform LDA by calculating the Cholesky factor of Sw and transforming the data
K = chol(Sw);
A = (K' \ Sb) / K;
[W, D] = eig(A);
Q = inv(K) * W;
%we project seperately 

LDA1 = Q(:, 1:2)' * X1;
LDA2 = Q(:, 1:2)' * X2;
LDA3 = Q(:, 1:2)' * X3;

% Visualize the LDA components
figure()
histogram(LDA1)
hold on
histogram(LDA2)
histogram(LDA3)
hold off
title("Components for LDA")

figure()
plot(LDA1(1,:), LDA1(2,:), 'r.', 'MarkerSize', 15)
hold on
plot(LDA2(1,:), LDA2(2,:), 'y.', 'MarkerSize', 15)
plot(LDA3(1,:), LDA3(2,:), 'g.', 'MarkerSize', 15)
hold off
set(gca, 'FontSize', 20)
title('LDA', 'FontSize', 20)

% Perform PCA by centering the data and calculating the singular value decomposition
x_means = 1/size(X, 2) * sum(X, 2);
X_centered = X - x_means * ones(1, size(X, 2));
[U, S, V] = svd(X_centered);
Z = U(:, 1:2)' * X_centered;

% Visualize the PCA components
figure()
plot(Z(1,:), Z(2,:), 'r.', 'MarkerSize', 15)
title("PCA")
xlabel('First principal component')
ylabel('Second principal component')
