% create random training set for 
% multivariate linear regression test

% d-dimensional m vector
d = 20;
m = 10 * randn([1 d]);
b = randn([1 1]);

% create N lines of training data
N = 1000000;
X = 10*randn([N d]);
Y = zeros([N 1]);

for i = 1:N
    Y(i,:) = m*X(i,:)' + b;
end

% horizontally concatenate X and Y
Z = horzcat(X,Y);
dlmwrite('multivariate_line_data.txt',Z,' ');