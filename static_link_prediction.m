% predictor for links - adj matrix 
% All node pairs - NP 
% distances between all node pairs - Dnp : X
% Existance of link (0/1) for corresponding node pairs : Y

% non linear predictor : logistic classifier/ SVM?

clc;
clear;

%% Prepare data:

VC = load('VC1.txt'); 
D = load('Original_Dist_nw1.txt'); 
[N,M] = size(VC);
%VC is taken from anchors .. not complete distance matrix.. although we can
%always take that

% get adjacency matrix 
A = zeros(N,N);
for i=1:N
    for j=1:N
        if D(i,j)==1 || i==j
            A(i,j) = 1;
        end
    end
end



p=1;
X = [];
Y = [];
for i=1:N
    for j=1:N
        if i==j
            X(p,:) = zeros(1,M);
            Y(p) = 1;
            p = p+1;
        else
            X(p,:) = VC(i,:)-VC(j,:);
            Y(p) = A(i,j);
            p = p+1;
            disp(i);
        end
    end
end

Y = Y';


%% randomly select node pairs for training
indices = randsample(length(Y),50000);
train_X = X(indices,:);
train_Y = Y(indices);

indices = randsample(length(Y),1000);
test_X = X(indices,:);
test_Y = Y(indices);


%% SVM training-RBF
% set the template with the required kernel function
t = templateSVM('KernelFunction','gaussian');
optpara = {'Coding', 'BoxConstraint', 'KernelScale'};
model = fitcecoc(train_X, train_Y,'Coding','onevsall','Learners',t, 'optimizeHyperparameters', optpara);
% or we can do 'all' to optimize all eligible parameters


%% normal SVM

model = fitcsvm(train_X, train_Y,'Standardize',false,'KernelFunction','RBF');

%% performance evaluation
[P_label,score] = predict(model,test_X);

len = length(P_label);


%% Performance evaluation:
test_Y = grp2idx(YTest);
P_label = grp2idx(YPred);

% %error and %accuracy
err = sum(abs(a-b))/numel(YTest);
err = err*100;
accuracy = 100-err

% precision and recall

[TP, FP, TN, FN] = calError(test_Y, P_label);

%Precesion & Recall:
Precesion = (TP)/(TP+FP)
Recall = (TP)/(TP+FN)

%F1 score:
F1 = (2*Precesion* Recall)/(Precesion+Recall);
