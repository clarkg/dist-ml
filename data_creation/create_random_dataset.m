% create line data

m = 2;
b = 3;

size_of_set = 100000;
x = 0:size_of_set;
y = m * x + b - randn([1 size_of_set+1]);

x = x';
y = y';

z = [x y];
dlmwrite('line_data.txt',z,' ');