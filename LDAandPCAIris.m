clear
close all

load IrisDataAnnotated.mat
I1= find(I==1);
I2=find(I==2);
I3= find(I==3);



% Split the dataset into three subsets based on class labels

X1=X(:,I1);
X2=X(:,I2);
X3=X(:,I3);
X=[X1 X2 X3];

% Calculate the class means and center the data within each class

p1= length(I1);
c1= 1/p1*sum(X1,2);
X1c= X1-c1*ones(1,p1);

p2= length(I2);
c2= 1/p2*sum(X2,2);
X2c= X2-c2*ones(1,p2);

p3= length(I3);
c3= 1/p3*sum(X3,2);
X3c= X3-c3*ones(1,p3);
% Calculate the within-class scatter matrix (Sw) and between-class scatter matrix (Sb)

Sw= X1c*X1c' + X2c*X2c' + X3c*X3c';
p= p1+p2+p3;
c=1/p*sum(X,2);

Sb=p1*(c1-c)*(c1-c)'+p2*(c2-c)*(c2-c)'+p3*(c3-c)*(c3-c)';

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
LDA1= Q(:,1:2)'*X1;
LDA2=Q(:,1:2)'*X2;
LDA3=Q(:,1:2)'*X3;
% Visualize the LDA components

figure()
histogram(LDA1)
hold on
histogram(LDA2)
histogram(LDA3)
hold off
title('LDA components')

figure()
plot(LDA1(1,:),LDA1(2,:),'r.','MarkerSize',15)
hold on
plot(LDA2(1,:),LDA2(2,:),'g.','MarkerSize',15)
plot(LDA3(1,:),LDA3(2,:),'b.','MarkerSize',15)
hold off
set(gca,'FontSize',20)
title('LDA projection','FontSize',20)
% Load the Iris dataset
load fisheriris;
X = meas;

% Calculate the mean of each feature
mean_X = mean(X);

% Center the data
X_centered = X - mean_X;

% Calculate the covariance matrix
covariance_matrix = cov(X_centered);

% Calculate the eigenvectors and eigenvalues
[eigenvectors, eigenvalues] = eig(covariance_matrix);

% Sort the eigenvalues in descending order
[eigenvalues, idx] = sort(diag(eigenvalues), 'descend');
eigenvectors = eigenvectors(:, idx);

% Determine the number of principal components to retain
total_variance = sum(eigenvalues);
explained_variance = eigenvalues / total_variance;
cumulative_variance = cumsum(explained_variance);
desired_explained_variance = 0.95;
num_components = find(cumulative_variance >= desired_explained_variance, 1);

% Select the top k eigenvectors based on the number of components
top_eigenvectors = eigenvectors(:, 1:num_components);

% Transform the data into the new feature space
X_transformed = X_centered * top_eigenvectors;


% Plotting the first two principal components
figure;
gscatter(X_transformed(:,1), X_transformed(:,2), species);
xlabel('1st Principal Component');
ylabel('2nd Principal Component');
title('PCA: Iris Dataset');
