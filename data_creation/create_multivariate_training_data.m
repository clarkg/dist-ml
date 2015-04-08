% create random training set for 
% multivariate linear regression test

function [m,b] = create_multivariate_training_data(d,N)

% d-dimensional m vector, b
% write both to disk
m = 10 * randn([1 d]);
b = randn([1 1]);
params = horzcat(m,b);
params_filename = sprintf('multivariate_line_params_d%d_n%d.txt',d,N);
dlmwrite(params_filename,params,' ');

% prepare data file for writing
data_filename = sprintf('multivariate_line_data_d%d_n%d.txt',d,N);

% create N lines of training data
% (forget each line after writing to disk)
for i = 1:N
    X = 10*randn([1 d]);
    Y = zeros([1 1]);
    Y = m*X' + b;
    Z = horzcat(X,Y);
    dlmwrite(data_filename,Z,'-append','delimiter',' ');
end