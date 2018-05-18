% new facebook network:

load ('nodepairsFacebook.mat');

XTrain = cell(100,1);

for pointer = 1:100
    h2 = WattsStrogatz(100,25,0.15);
    G = h2;

    features = zeros(3,1000);
    % compute features for each node pair
     
    
    [l, col] = size(E(:,:));
    for i=1:l
        vv = E.EndNodes(i,:);
        v1 = vv(1);
        v2 = vv(2);
        d1 = degree(G, v1); 
        d2 = degree(G, v2); 
        nebr1 = neighbors (G,v1);
        nebr2 = neighbors (G,v2);
        common = numel(intersect(nebr1,nebr2));
        clos = C1(v1) * C1(v2);
        page = C2(v1) * C2(v2); 
        Eg = C3(v1) * C3(v2);
        pr = d1*d2;
        J = common/numel((union(nebr1,nebr2)));
        So = 2*J/(d1+d2);
        features(1,i) = common;
         
        features(2,i) = pr/1000;
        features(3,i) = J*5;
         
    end
    XTrain{pointer} = features;
    
end

filename = 'inputLSTM.mat';
save(filename);

load ( 'inputLSTM.mat');
YTrain = randi([0 1],1000,1);


figure
plot(XTrain{1}')
xlabel("Time Step")
set(gcf,'color','w');
title("Training Observation 1 - dynamic network features")
legend("Feature " + string(1:7),'Location','northeastoutside')


numObservations = numel(XTrain);
for i=1:numObservations
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths);
XTrain = XTrain(idx);
YTrain = YTrain(idx);

figure
bar(sequenceLengths)
ylim([0 30])
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")

miniBatchSize = 27;

inputSize = 12;
numHiddenUnits = 100;
numClasses = 9;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

maxEpochs = 100;
miniBatchSize = 27;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);

% testing-------------

[XTest,YTest] = japaneseVowelsTestData;
XTest(1:3)

numObservationsTest = numel(XTest);
for i=1:numObservationsTest
    sequence = XTest{i};
    sequenceLengthsTest(i) = size(sequence,2);
end
[sequenceLengthsTest,idx] = sort(sequenceLengthsTest);
XTest = XTest(idx);
YTest = YTest(idx);

miniBatchSize = 27;
YPred = classify(net,XTest, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');

acc = sum(YPred == YTest)./numel(YTest)
