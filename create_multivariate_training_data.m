% create random training set for 
% multivariate linear regression test

% d-dimensional m vector
d = 10;
m = 10 * randn([1 d]);
b = randn([1 1]);

% create N lines of training data
%N = 100000000;
N = 10;
X = 10*randn([d N]);
Y = zeros([N 1]);

for i = 1:N
    Y(i,:) = m*X(i,:)';
end

% horizontally concatenate X and Y
Z = horzcat(X,Y);
dlmwrite('multivariate_line_data.txt',Z,' ');