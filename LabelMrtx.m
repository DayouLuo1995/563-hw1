function [Image_sub, label] = LabelMrtx(Image, Label, N)
%%Image is the image data, which is N *784
%%Label is an array, with N * 1
%%N is the amount of image that included in the data set.
%%the output of this function is a dataset of images,and a matrix of
%%labels. 1 as (1 0 0 ...0)
Image_sub = Image(1:N,:);
Trainlbl_small = Label(1:N,:);
Trainlbl_sm_mat = zeros(N, 10);
for iter = 1:10
    tab = find(Trainlbl_small == mod(iter, 10));
    Trainlbl_vector = zeros(1,10);
    Trainlbl_vector(iter) = 1;
    for iter2 = tab'
        Trainlbl_sm_mat(iter2, :) = Trainlbl_vector;
    end
end
label = Trainlbl_sm_mat;
end