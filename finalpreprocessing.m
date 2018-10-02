% preprocessing for all images

% image process
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables. Or clearvars if you want.

DatasetPath = fullfile('/Volumes/UNTITLED/SigComp2011/trainingSet/OfflineSignatures/Chinese/TrainingSet/Offline Genuine');
imds = imageDatastore(DatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
N=size(imds.Files,1);
for k=1:N
     imds.Files{k,1}=erase(imds.Files{k,1},"._");
     str_array=strsplit(imds.Files{k,1},'/');
     str=str_array(size(str_array,2));
    img=imread(erase(imds.Files{k,1},"._"));
    grayim=rgb2gray(img);
    Nredim=medfilt2(grayim);
    %Binim=imbinarize(Nredim);
    Resizeim=imresize(Nredim,[64,128]);
    SavesetPath = strcat('/Volumes/UNTITLED/SigComp2011(resize 64X128/Chinese/1/',str);
    imwrite(Resizeim,SavesetPath{1});
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% preprocessing for all images

% image process
% clc;    % Clear the command window.
% close all;  % Close all figures (except those of imtool.)
% clear;  % Erase all existing variables. Or clearvars if you want.
% for k=7:7
% N = sprintf( '%04d', k );
% OrigImagefilename=strcat('/Users/jerry/Downloads/Pictures/IMG_20180706_',N,'.png');
% f=imread(OrigImagefilename);
% rect=[270,930,2159,2323];
% cropimg=imcrop(f,rect);
% imshow(cropimg);
% hold on
% ij=1;
% for i=1:7 
%     for j=1:3
%         orgim=imcrop(cropimg);
%         grayim=rgb2gray(orgim);
%         Nredim=medfilt2(grayim);
%         Binim=imbinarize(Nredim);
%         Resizeim=imresize(Binim,[64,128]);
%         imwrite(Resizeim,strcat('/Volumes/UNTITLED/SigReading2018(resize 64X128/',N,'_',num2str(ij),'.png'));
%         ij=ij+1;
%     end
% end
% end






