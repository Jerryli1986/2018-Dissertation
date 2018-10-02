% signature verification statistics
  clc;
  clear;

    
    numTrainFiles =0.7;
    TrainingDatasetPath='/Volumes/UNTITLED/SigReading2018(resize 64X128'; %/Volumes/UNTITLED/SigComp2011(resize 64X128/Chinese';
    %'/Volumes/UNTITLED/SigReading2018(resize 64X128';
    TestingDatasetPath='';
    %TestingDatasetPath='/Volumes/UNTITLED/SigReading2018(resize 64X128';
    acc=[];
    for Button=5:5
        accuracy_array=[];
    for x=1:20
        if (strcmp(TestingDatasetPath,''))
            singletest=0;
            DatasetPath = fullfile(TrainingDatasetPath);
            imds = imageDatastore(DatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
            N=size(imds.Files,1);
            for j=1:N
                imds.Files{j,1}=erase(imds.Files{j,1},"._");
            end
            [imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
        else
            Train_file = fullfile(TrainingDatasetPath);
            imdsTrain = imageDatastore(Train_file,'IncludeSubfolders',true,'LabelSource','foldernames');
            N1=size(imdsTrain.Files,1);
            for j=1:N1
                imdsTrain.Files{j,1}=erase(imdsTrain.Files{j,1},"._");
            end
            Test_file = fullfile(TestingDatasetPath);
            imdsValidation= imageDatastore(Test_file,'IncludeSubfolders',true,'LabelSource','foldernames');
            N2=size(imdsValidation.Files,1);
            for k=1:N2
                imdsValidation.Files{k,1}=erase(imdsValidation.Files{k,1},"._");
            end

            if (size(imdsValidation.Files,1)==1)
               singletest=1; 
            else
               singletest=0;
            end
        end
        if (Button==1)
            TrainMat=img2Mat(imdsTrain);
            Meanface=mean(TrainMat,2);
            ShiftTrainMat=double(TrainMat)-Meanface;
            [coeff,~,~,~,explained]=pca(ShiftTrainMat);
            
            % number of eigenvector who totally explainedratio>95%
            explainedratio=0;
            m=0;
            while explainedratio<=95
                m=m+1;
                explainedratio=explainedratio+explained(m);
            end
            eigenfaces=ShiftTrainMat*coeff(:,1:m);
            W_TrainMat=inv(eigenfaces'*eigenfaces)*(eigenfaces')*ShiftTrainMat; % each column for each image
            
            TestMat=img2Mat(imdsValidation);
            ShiftTestMat=double(TestMat)-Meanface;
            W_TestMat=inv(eigenfaces'*eigenfaces)*(eigenfaces')*ShiftTestMat;
            
            YTrain = imdsTrain.Labels;
            YValidation = imdsValidation.Labels;
            
            % Distance classifier
            YPred=categorical([]);
            for i=1:size(W_TestMat,2)
                euclide_dist = [];
                for j=1 : size(W_TrainMat,2)
                    temp = norm(W_TestMat(:,i)-W_TrainMat(:,j))^2;
                    euclide_dist = [euclide_dist temp];
                end
                [euclide_dist_min, recognized_index] = min(euclide_dist);
                YPred=[YPred;YTrain(recognized_index)];
            end
            Accuracy_x=output(singletest,YValidation,YPred);
            %displayimg(singletest,imdsTrain,YPred);
        elseif (Button==2)
            TrainMat=img2Mat(imdsTrain);
            Meanface=mean(TrainMat,2);
            ShiftTrainMat=double(TrainMat)-Meanface;
            [coeff,~,~,~,explained]=pca(ShiftTrainMat);
            % number of eigenvector who totally explainedratio>95%
            explainedratio=0;
            m=0;
            while explainedratio<=95
                m=m+1;
                explainedratio=explainedratio+explained(m);
            end
            eigenfaces=ShiftTrainMat*coeff(:,1:m);
            W_TrainMat=inv(eigenfaces'*eigenfaces)*(eigenfaces')*ShiftTrainMat;
            
            TestMat=img2Mat(imdsValidation);
            ShiftTestMat=double(TestMat)-Meanface;
            W_TestMat=inv(eigenfaces'*eigenfaces)*(eigenfaces')*ShiftTestMat;
            
            YTrain = imdsTrain.Labels;
            YValidation = imdsValidation.Labels;
            
            % SVM classifer
            classifier = fitcecoc(W_TrainMat',YTrain);
            YPred = predict(classifier,W_TestMat');
            
            Accuracy_x=output(singletest,YValidation,YPred);
            %displayimg(singletest,imdsTrain,YPred)
        elseif (Button==3)
            N_class=size(unique(imdsTrain.Labels),1);
            % define CNN network architecture
            layers = [
                imageInputLayer([64 128 1])
                
                convolution2dLayer(3,16,'Padding',1)
                batchNormalizationLayer
                reluLayer
                
                maxPooling2dLayer(2,'Stride',2)
                
                convolution2dLayer(3,32,'Padding',1)
                batchNormalizationLayer
                reluLayer
                
                maxPooling2dLayer(2,'Stride',2)
                
                convolution2dLayer(3,64,'Padding',1)
                batchNormalizationLayer
                reluLayer
                
                maxPooling2dLayer(2,'Stride',2)
                
                convolution2dLayer(3,64,'Padding',1)
                batchNormalizationLayer
                reluLayer
                
                maxPooling2dLayer(2,'Stride',2)
                
                
                
                
                fullyConnectedLayer(N_class)
                softmaxLayer
                classificationLayer];
            
            % Specify training options
            %'InitialLearnRate',0.05,...
            options = trainingOptions('sgdm', ...
                'MaxEpochs',50, ...
                'Verbose',false, ...
                'Plots','training-progress');
            % train Network using traning data
            net = trainNetwork(imdsTrain,layers,options);
            
            % Classify Validation images and accuracy
            YPred = classify(net,imdsValidation);
            YValidation = imdsValidation.Labels;
            Accuracy_x=output(singletest,YValidation,YPred);
            %displayimg(singletest,imdsTrain,YPred)
        elseif (Button==4)
            % define CNN network architecture
            N_class=size(unique(imdsTrain.Labels),1);
            layers = [
                imageInputLayer([64 128 1])
                
                convolution2dLayer(3,16,'Padding',1)
                batchNormalizationLayer
                reluLayer
                
                maxPooling2dLayer(2,'Stride',2)
                
                convolution2dLayer(3,32,'Padding',1)
                batchNormalizationLayer
                reluLayer
                
                maxPooling2dLayer(2,'Stride',2)
                
                convolution2dLayer(3,64,'Padding',1)
                batchNormalizationLayer
                reluLayer
                
                maxPooling2dLayer(2,'Stride',2)
                
                convolution2dLayer(3,64,'Padding',1)
                batchNormalizationLayer
                reluLayer
                
                maxPooling2dLayer(2,'Stride',2)
                
                
                
                
                fullyConnectedLayer(N_class)
                softmaxLayer
                classificationLayer];
            
            % Specify training options
            %'InitialLearnRate',0.05,...
            options = trainingOptions('sgdm', ...
                'MaxEpochs',50, ...
                'Verbose',false, ...
                'Plots','training-progress');
            % train Network using traning data
            net = trainNetwork(imdsTrain,layers,options);
            
            % features extraction
            layer = 'fc';
            featuresTrain = activations(net,imdsTrain,layer,'OutputAs','rows');
            featuresTest = activations(net,imdsValidation,layer,'OutputAs','rows');
            YTrain = imdsTrain.Labels;
            YValidation = imdsValidation.Labels;
            % classifer
            %  Use the features extracted from the training images as predictor
            %  variables and fit a multiclass support vector machine (SVM) using
            %  fitcecoc
            
            classifier = fitcecoc(featuresTrain,YTrain);
            YPred = predict(classifier,featuresTest);
            Accuracy_x=output(singletest,YValidation,YPred);
            %displayimg(singletest,imdsTrain,YPred)
        elseif (Button==5)
            YTrain = imdsTrain.Labels;
            YValidation = imdsValidation.Labels;
            
            Trainbag = bagOfFeatures(imdsTrain);
            %trainfeaturevector=encode(bag, img);
            % SVM classifer
            classifier =trainImageCategoryClassifier(imdsTrain,Trainbag);
            
            if (singletest==1)
                
                img = imread(imdsValidation.Files{1,1});
                YPred= predict(classifier, img);
                YPred=categorical(YPred);
                accuracy=sum(YPred == YValidation)/numel(YValidation);
                if (accuracy==1)
                    results='This is valid signature';
                else
                    results='This is invalid signature';
                end
            else
                YPred=categorical([]);
                confMatrix = evaluate(classifier,imdsValidation);
                accuracy=mean(diag(confMatrix));
                results=strcat('The accuracy is :',num2str(round(accuracy,4)*100),'%');
            end
            Accuracy_x=accuracy;
            %displayimg(singletest,imdsTrain,YPred)
        elseif (Button==6)
            img = readimage(imdsTrain, 1);
            [HOGfeaturevec,~]=extractHOGFeatures(img,'CellSize',[4 4]);
            cellSize = [4 4];
            hogFeatureSize=length(HOGfeaturevec);
            trainingFeatures=HOG_feature(imdsTrain,hogFeatureSize,cellSize);
            
            YTrain = imdsTrain.Labels;
            YValidation = imdsValidation.Labels;
            % SVM classifer
            classifier = fitcecoc(trainingFeatures, YTrain);
            testFeatures=HOG_feature(imdsValidation,hogFeatureSize,cellSize);
            YPred = predict(classifier,testFeatures);
            Accuracy_x=output(singletest,YValidation,YPred);
            %displayimg(singletest,imdsTrain,YPred)
        end
        accuracy_array=[accuracy_array,Accuracy_x];
        x
    end
    %accuracy_average=mean(accuracy_array)
    Button
    acc=[acc;accuracy_array]
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function TrainMat=img2Mat(imdsTrain)
            TrainMat=[];
            for i = 1 :size(imdsTrain.Files,1)
                img = imread(imdsTrain.Files{i,1});    %img is gray if not need to rgb2gray(img)
                [r ,c] = size(img);
                temp = reshape(img',r*c,1);  %% Reshaping 2D images into 1D image vectors
                TrainMat = [TrainMat temp];
            end
        end
        
        function Features=HOG_feature(imdsTrain,hogFeatureSize,cellSize)
            numImages = numel(imdsTrain.Files);
            Features = zeros(numImages, hogFeatureSize, 'single');
            % preprocess
            for i = 1:numImages
                img = readimage(imdsTrain, i);
                %img = rgb2gray(img);
                % Apply pre-processing steps
                img = imbinarize(img);
                
                Features(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
            end
        end
        function accuracy=output(singletest,YValidation,YPred)
            accuracy=sum(YPred == YValidation)/numel(YValidation);
            if (singletest==1)
                if (accuracy==1)
                    results='This is valid signature';
                else
                    results='This is invalid signature';
                end
            else
                results=strcat('The accuracy is : ',num2str(round(accuracy,4)*100),'%');
            end
        end
        function displayimg(singletest,imdsTrain,YPred)
            if (singletest==1)
                indexPred=find(ismember(imdsTrain.Labels, YPred),1, 'first');
                imgPred=imread(imdsTrain.Files{indexPred,1});
                imshow(imgPred);
            end
        end
        % 1) PCA+Euclidean Distance classifer
        % 2) PCA+SVM
        % 3) CNN+softmax
        % 4) CNN+SVM
        % 5) BOW+SVM
        % 6) HOG+SVM
    