close all
clc
%%%%%%%  provide the data path where the training images are present  %%%%%%%
% datapath = 'give here the path of your training images';
% testpath = 'similarly give the path for test images';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
datapath = uigetdir('C:\Users\Shaq\Desktop\will smith','*.jpg');
testpath = uigetdir('C:\Users\Shaq\Desktop\will smith 2');

prompt = {'Enter test image name (a number between 1 to 10):'};
dlg_title = 'Input of PCA-Based Face Recognition System';
num_lines= 1;
def = {' '};
TestImage = inputdlg(prompt,dlg_title,num_lines,def);
TestImage = strcat(testpath,'\',char(TestImage),'.jpg');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%  calling the functions  %%%%%%%%%%%%%%%%%%%%%%%%

recog_img = facerecog(datapath,TestImage);
selected_img = strcat(datapath,'\',recog_img);
select_img = imread(selected_img);
imshow(select_img);
title('Recognized Image');
test_img = imread(TestImage);
figure,imshow(test_img);
title('Test Image');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result = strcat('the recognized image is : ',recog_img);
disp(result);


