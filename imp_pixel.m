function N = imp_pixel(X, M, option)
%%X is the outcome of regression, option is a scaler. 
%can be 1 or 2. 1for the pixel universually important,
% 2 for the pixel important for certain digits.
%N is a vector or a matrix, corresponding the options
%If N is a matrix, it should be M * 10, where M is number of pixel choosen

if option == 1
    %pick pixcel for all digits
    N = sum(abs(X), 2);%row sums
    [~, N] = maxk(N, M);
    return
else 
    %for each digits
    [~, N] = maxk(abs(X), M);
end
end
