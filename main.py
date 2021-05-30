import os
import sys
import h5py
import numpy
import tensorflow
from PIL import Image
from enum import IntEnum
import matplotlib.pyplot
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

DATASET_IMAGE_RESOLUTION = (900, 900)
RES_NET_50_INPUT_RESOLUTION = (224, 224)
RES_NET_50_INPUT_SHAPE = RES_NET_50_INPUT_RESOLUTION + (3,)
CHECKPOINT_PATH = "bestModel.h5"

class Label(IntEnum):
    BENIGN_MASS = 0
    BENIGN_CALCIFICATION = 1
    MALIGNANT_MASS = 2
    MALIGNANT_CALCIFICATION = 3

def prepareLabels(labels):
    return labels % 2

def prepareImage(imageArray):
    image = Image.fromarray(imageArray.astype(numpy.float32))
    image = image.resize(RES_NET_50_INPUT_RESOLUTION)
    image = image.convert("RGB")
    return preprocess_input(numpy.array(image))

def getDataset(skipTraining):
    print("Reading Data ...")

    if not skipTraining:
        trainingMetaInfo = h5py.File("dataset/train_meta.h5", "r")
        numberOfTrainingImages = trainingMetaInfo["data"]["axis1"].shape[0]
        rawTrainingData = numpy.array(numpy.memmap("dataset/train_data.npy", dtype = "uint8", mode = "r",
                                      shape = (numberOfTrainingImages, *DATASET_IMAGE_RESOLUTION)))
        rawTrainingLabels = numpy.array(numpy.memmap("dataset/train_labels.npy", dtype = "uint8",
                                        mode = "r", shape = (numberOfTrainingImages,)))
    else:
        rawTrainingData = None
        rawTrainingLabels = None

    testingMetaInfo = h5py.File("dataset/test_meta.h5", "r")
    numberOfTestingImages = testingMetaInfo["data"]["axis1"].shape[0]
    rawTestingData = numpy.array(numpy.memmap("dataset/test_data.npy", dtype = "uint8", mode = "r",
                                 shape = (numberOfTestingImages, *DATASET_IMAGE_RESOLUTION)))
    rawTestingLabels = numpy.array(numpy.memmap("dataset/test_labels.npy", dtype = "uint8",
                                   mode = "r", shape = (numberOfTestingImages,)))

    return rawTrainingData, rawTestingData, rawTrainingLabels, rawTestingLabels

def prepareDataset(rawTrainingData, rawTestingData, rawTrainingLabels, rawTestingLabels, skipTraining):
    print("Preparing Data ...")

    if not skipTraining:
        orderedTrainingData = numpy.array([prepareImage(imageArray) for imageArray in rawTrainingData])
        orderedTrainingLabels = prepareLabels(rawTrainingLabels)
        trainingPermutation = numpy.random.permutation(rawTrainingData.shape[0])
        trainingData = orderedTrainingData[trainingPermutation]
        trainingLabels = orderedTrainingLabels[trainingPermutation]
    else:
        trainingData = None
        trainingLabels = None

    orderedTestingData = numpy.array([prepareImage(imageArray) for imageArray in rawTestingData])
    orderedTestingLabels = prepareLabels(rawTestingLabels)
    testingPermutation = numpy.random.permutation(rawTestingData.shape[0])
    testingData = orderedTestingData[testingPermutation]
    testingLabels = orderedTestingLabels[testingPermutation]

    return trainingData, testingData, trainingLabels, testingLabels

def getTrainingModel():
    print("Preparing Model ...")
    baseModel = ResNet50(include_top = False, input_shape = RES_NET_50_INPUT_SHAPE, pooling = "avg")
    baseModel.trainable = False
    lastLayerOutput = baseModel.layers[-1].output
    predictionLayerOutput = Dense(1, activation = sigmoid, name = "predictions")(lastLayerOutput)
    predictionModel = Model(inputs = baseModel.input, outputs = predictionLayerOutput)
    predictionModel.compile(optimizer = Adam(), loss = BinaryCrossentropy(from_logits = True),
                            metrics = [BinaryAccuracy()])

    return predictionModel

def trainModel(model, trainingData, trainingLabels):
    print("Training Model ...")
    earlyStopping = EarlyStopping(monitor = "val_binary_accuracy", mode = "max", verbose = 1,
                                  patience = 5)
    modelCheckpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor = "val_binary_accuracy", mode = "max",
                                      verbose = 1, save_best_only = True)
    callbacks = [earlyStopping, modelCheckpoint]
    model.fit(trainingData, trainingLabels, callbacks = callbacks, validation_split = 0.2,
              epochs = 20, batch_size = 16)

def getClassActivationMapModel(baseModel):
    print("Preparing Class Activation Map Model ...")
    lastConvolutionLayer = baseModel.get_layer("conv5_block3_out").output
    return Model(inputs = model.input, outputs = lastConvolutionLayer)

def getPredictionWeights(model):
    return model.get_layer("predictions").get_weights()[0].squeeze()

def showClassActivationMap(model, predictionWeights, imageArray):
    print("Predicting Sample Image ...")
    image = prepareImage(imageArray)
    activationMaps = model.predict(image[numpy.newaxis,:]).squeeze()
    activationMap = numpy.dot(activationMaps.reshape((7 * 7, -1)), predictionWeights).reshape(7, 7)

    figure, axes = matplotlib.pyplot.subplots(1, 2)
    axes[0].imshow(imageArray, interpolation = "bicubic")
    axes[1].imshow(activationMap, interpolation = "bicubic")
    matplotlib.pyplot.show()

########
# Driver
########

skipTraining = "--skip-training" in sys.argv and os.path.isfile(CHECKPOINT_PATH)

rawTrainingData, rawTestingData, rawTrainingLabels, rawTestingLabels = getDataset(skipTraining)
trainingData, testingData, trainingLabels, testingLabels = prepareDataset(
        rawTrainingData, rawTestingData, rawTrainingLabels, rawTestingLabels, skipTraining)

model = getTrainingModel()
if skipTraining:
    model.load_weights(CHECKPOINT_PATH)
else:
    trainModel(model, trainingData, trainingLabels)

classActivationMapModel = getClassActivationMapModel(model)
predictionWeights = getPredictionWeights(model)
showClassActivationMap(classActivationMapModel, predictionWeights, rawTestingData[0])
