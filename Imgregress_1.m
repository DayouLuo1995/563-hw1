function X = Imgregress_1(img, label, option)
%%img is a the training data of imgs, label is the label matrix
%%option can be 'lasso', 'pinv', 'backslash'

 LX = 784;% the size of an image

 if strcmp(option, 'pinv')
     X = pinv(img) * label;
     return;
 end
 if strcmp(option, 'backslash')
     X = img\label;
     return;
 end
 if strcmp(option, 'lasso')
     %do lasso regression for each iteration
     X = zeros(LX, 10);
     for ii = 1:10
         [X_temp, Fitinfo]= lasso(img, label(:,ii), 'CV', 3);%%Lasso regression for the ii column
          idxLambdaMMSE = Fitinfo.IndexMinMSE;
          disp(Fitinfo.Lambda(idxLambdaMMSE));
           X(:,ii)= X_temp(:,idxLambdaMMSE);
     end
     return;
 else
     error('Wrong Option')
 end

