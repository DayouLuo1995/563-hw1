
% %%Assemble
 clear all;
 close all;
 clc;
 load('Data')

N = 60000;%Use all training data
[TImg, Tlabel] = LabelMrtx(TrainImage, TrainLabel, N);

%%First regression
X_BS= Imgregress_1(TImg, Tlabel, 'backslash');
X_pinv = Imgregress_1(TImg, Tlabel, 'pinv');
X_lasso = Imgregress_1(TImg, Tlabel, 'lasso');
%%Get the most important pixels.

N = [10, 20, 40, 80, 120, 200, 300, 400, 500, 784];
LN = 10;
R = zeros(11, LN, 3);%The accuracy rates.
M = 10000;%sample size of the test data
Test_Img = TestImage(1:M, :);
Test_lbl = TestLabel(1:M);

%whole picturewise

for ii = 1:LN
    Pixels = imp_pixel(X_BS, N(ii), 1);
    %X_BS_new = Imgregress(TImg, Tlabel,'backslash',Pixels);
    R(:,ii,1) = imgtest(Test_Img, Test_lbl, X_BS, Pixels);
    %
    Pixels = imp_pixel(X_pinv, N(ii), 1);
 %   X_pinv_new = Imgregress(TImg, Tlabel, 'pinv', Pixels);
    R(:,ii, 2) = imgtest(Test_Img, Test_lbl, X_pinv, Pixels);
    %
     Pixels = imp_pixel(X_lasso, N(ii), 1);
    %X_lasso_new = Imgregress(TImg, Tlabel, 'lasso', Pixels);
    R(:,ii, 3) = imgtest(Test_Img, Test_lbl, X_lasso, Pixels);
end
%%Each digits
R_digits = zeros(11, LN, 3);
for ii = 1:LN
    Pixels = imp_pixel(X_BS, N(ii), 2);
    %X_BS_new = Imgregress(TImg, Tlabel,'backslash',Pixels);
    R_digits(:,ii,1) = imgtest(Test_Img, Test_lbl, X_BS, Pixels);
    %
    Pixels = imp_pixel(X_pinv, N(ii), 2);
    %X_pinv_new = Imgregress(TImg, Tlabel, 'pinv', Pixels);
    R_digits(:,ii, 2) = imgtest(Test_Img, Test_lbl, X_pinv, Pixels);
    %
     Pixels = imp_pixel(X_lasso, N(ii), 2);
    %X_lasso_new = Imgregress(TImg, Tlabel, 'lasso', Pixels);
    R_digits(:,ii, 3) = imgtest(Test_Img, Test_lbl, X_lasso, Pixels);
end
%As the scale changed after we picked digits, we used a linear regression
%to rescale the data
R_rescale = zeros(11, LN, 3);
for ii = 1:LN
    Pixels = imp_pixel(X_BS, N(ii), 1);
    %X_BS_new = Imgregress(TImg, Tlabel,'backslash',Pixels);
    R_rescale(:,ii,1) = imgtest_rescale(Test_Img, Test_lbl,Pixels, TImg, Tlabel);
    %
    Pixels = imp_pixel(X_pinv, N(ii), 1);
    %X_pinv_new = Imgregress(TImg, Tlabel, 'pinv', Pixels);
    R_rescale(:,ii, 2) = imgtest_rescale(Test_Img, Test_lbl,Pixels, TImg, Tlabel);
    %
     Pixels = imp_pixel(X_lasso, N(ii), 1);
    %X_lasso_new = Imgregress(TImg, Tlabel, 'lasso', Pixels);
    R_rescale(:,ii, 3) =  imgtest_rescale(Test_Img, Test_lbl,Pixels, TImg, Tlabel);
end

%%
R_rescale_digits = zeros(11, LN, 3);
for ii = 1:LN
    Pixels = imp_pixel(X_BS, N(ii), 2);
    %X_BS_new = Imgregress(TImg, Tlabel,'backslash',Pixels);
    R_rescale_digits(:,ii,1) = imgtest_rescale(Test_Img, Test_lbl,Pixels, TImg, Tlabel);
    %
    Pixels = imp_pixel(X_pinv, N(ii), 2);
    %X_pinv_new = Imgregress(TImg, Tlabel, 'pinv', Pixels);
    R_rescale_digits(:,ii, 2) = imgtest_rescale(Test_Img, Test_lbl,Pixels, TImg, Tlabel);
    %
     Pixels = imp_pixel(X_lasso, N(ii), 2);
    %X_lasso_new = Imgregress(TImg, Tlabel, 'lasso', Pixels);
    R_rescale_digits(:,ii, 3) =  imgtest_rescale(Test_Img, Test_lbl,Pixels, TImg, Tlabel);
end


%%Plot
%%pcolor
A = find(X_lasso(:,1));
size(A);
B = zeros(1, 784);
B(A) = 1;
pcolor(reshape(B, 28,28));
%%
for ii= 1:10
    A = X_lasso(:,ii);
    [~,ia] = maxk(abs(A), 200);
    B = zeros(1, 784);
    B(ia) = 1;
    subplot(2,5,ii);
    colormap gray;
    pcolor(reshape(B,28,28)');
    
    h = ['digits ', num2str(mod(ii, 10))];
    title(h)
  
end
%%
for ii = 1:10
    a = find(Test_lbl==mod(ii, 10),1);
    colormap gray;
    subplot(2,5,ii)
    imagesc(reshape(Test_Img(a,:), 28,28)')
    h = ['digits ', num2str(mod(ii, 10))];
    title(h)
end
%%
%%Rate
figure(4);
hold on;
for i = 1:3
 plot(N, R(11,:,i),'LineWidth',2);   
plot(N, R_rescale(11,:,i), 'LineWidth',2);
end

legend('Back Slash', 'Rescaled Black Slash', 'Pseudo-inverse', 'Rescaled Pseudo-inverse', 'Lasso', 'Rescaled Lasso');


%%
%correctness for each digits:
figure(5)
hold on
for ii= 1:10
    subplot(2,5,ii);

    for jj = 1:3
        hold on
        plot(N, R_rescale_digits(ii, :, jj)); 
        pause(0.5)
    end
    h = ['Digits ', num2str(mod(ii, 10))];
    title(h)
  legend('Back Slash', 'Pseudo-inverse', 'Lasso', 'Location', 'southeast');

end
 
