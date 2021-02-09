function lgraph = residualgraph(netWidth,numUnits)


% Check inputs
assert(numUnits > 0 && mod(numUnits,3) == 0 ,...
    "Number of convolutional units must be an integer multiple of 3.");
unitsPerStage = numUnits/3;

convolutionalUnit = @ConvolutionalUnit;

layers = [
    imageInputLayer([256 256 3],'Name','input') %input image size
    convolution2dLayer(3,netWidth,'Padding','same','Name','convInp')
    batchNormalizationLayer('Name','BNInp')
    reluLayer('Name','reluInp')];

% Stage one. Activation size is 32-by-32.
for i = 1:unitsPerStage
    layers = [layers
        convolutionalUnit(netWidth,1,['S1U' num2str(i) '_'])
        additionLayer(2,'Name',['add1' num2str(i)])
        reluLayer('Name',['relu1' num2str(i)])];
end

% Stage two. Activation size is 16-by-16.
for i = 1:unitsPerStage
    if i==1
        stride = 2;
    else
        stride = 1;
    end
    layers = [layers
        convolutionalUnit(2*netWidth,stride,['S2U' num2str(i) '_'])
        additionLayer(2,'Name',['add2' num2str(i)])
        reluLayer('Name',['relu2' num2str(i)])];
end

% Stage three. Activation size is 8-by-8
for i = 1:unitsPerStage
    if i==1
        stride = 2;
    else
        stride = 1;
    end
    layers = [layers
        convolutionalUnit(4*netWidth,stride,['S3U' num2str(i) '_'])
        additionLayer(2,'Name',['add3' num2str(i)])
        reluLayer('Name',['relu3' num2str(i)])];
end

% Output section.
layers = [layers
    averagePooling2dLayer(8,'Name','globalPool')
    fullyConnectedLayer(38,'Name','fcFinal') %we know there are extactly 38 categories of 38 different objects
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];

lgraph = layerGraph(layers);


%% Add shortcut connections
% Add shortcut connection around the convolutional units. Most shortcuts
% are identity connections.
for i = 1:unitsPerStage-1
    lgraph = connectLayers(lgraph,['relu1' num2str(i)],['add1' num2str(i+1) '/in2']);
    lgraph = connectLayers(lgraph,['relu2' num2str(i)],['add2' num2str(i+1) '/in2']);
    lgraph = connectLayers(lgraph,['relu3' num2str(i)],['add3' num2str(i+1) '/in2']);
end


lgraph = connectLayers(lgraph,'reluInp','add11/in2');

numF =  netWidth*2;

skip1 = [convolution2dLayer(1,numF,'Stride',2,'Name','skipConv1')
    batchNormalizationLayer('Name','skipBN1')];
lgraph = addLayers(lgraph,skip1);
lgraph = connectLayers(lgraph,['relu1' num2str(unitsPerStage)],'skipConv1');
lgraph = connectLayers(lgraph,'skipBN1','add21/in2');

% Shortcut connection from stage two to stage three.

numF =  netWidth*4;

skip2 = [convolution2dLayer(1,numF,'Stride',2,'Name','skipConv2')
    batchNormalizationLayer('Name','skipBN2')];
lgraph = addLayers(lgraph,skip2);
lgraph = connectLayers(lgraph,['relu2' num2str(unitsPerStage)],'skipConv2');
lgraph = connectLayers(lgraph,'skipBN2','add31/in2');

return


end

%%
% layers = standardConvolutionalUnit(numF,stride,tag) creates a standard
% convolutional unit, containing two 3-by-3 convolutional layers with numF
% filters and a tag for layer name assignment.
function layers = ConvolutionalUnit(numF,stride,tag)
layers = [
    convolution2dLayer(3,numF,'Padding','same','Stride',stride,'Name',[tag,'conv1'])
    batchNormalizationLayer('Name',[tag,'BN1'])
    reluLayer('Name',[tag,'relu1'])
    convolution2dLayer(3,numF,'Padding','same','Name',[tag,'conv2'])
    batchNormalizationLayer('Name',[tag,'BN2'])];
end
