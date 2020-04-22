function [R] = imgtest(TestImg, Testlbl, X, N)
%TestImg is the data used for testing, M*784
%Testlbl is a vector, containing all the labels for testIMG
%X is the outcome from the regression, here we assume X has 784 pixels.
%R is the rate of correctness, a 1*10 vector
%N is the pixel chosen for identification, N can be a vector or a matrix
%S is the rate of correctness, given the test img is fromt he same digits
R = zeros(11,1);

%
if size(N, 2)==1%N is the universal choosen pixel
    TestImg = TestImg(:, N);
    Test_temp = TestImg * X(N, :);
    [~, Test_result] = maxk(Test_temp, 1, 2);
    Test_result = mod(Test_result - Testlbl, 10);%Test_result =0 correct classification
    for ii = 1:10%Correctness rate
        index = Testlbl == mod(ii, 10);%Testlabel equals ii
        R(ii) = sum(Test_result(index)==0)/ sum(index);
    end
    R(11) = sum(Test_result == 0 )/ length(Test_result);
    return
%
%
else
    L = size(TestImg,1);
    [~, labelmtrix] = LabelMrtx(TestImg, Testlbl, L);
    for ii = 1:10
        Testimg_temp = TestImg(:, N(:,ii));
        Testlbl_temp = Testimg_temp*X(N(:,ii),ii);
        Test_result = zeros(L,1);
        Test_result(Testlbl_temp>=0.5) = 1;%assign the large numbers to the 
   %digit.
        Test_result = Test_result - labelmtrix(:, ii);
        index = labelmtrix(:,ii) == 1;
        R(ii) = 1 - sum(Test_result == -1)/ sum(index);%sum index is the number of 
        %ii in the test data and if regression gives a wrong prediction,
        %The testresult will be -1
    end
end
   

