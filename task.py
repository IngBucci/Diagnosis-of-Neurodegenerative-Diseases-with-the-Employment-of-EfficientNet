import os
import glob
import argparse
import numpy as np
import tensorflow
import matplotlib.pyplot as plt

import keras.optimizers

from tensorflow import keras
from keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, preprocess_input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import layers

from time import time

from sklearn.metrics import confusion_matrix, classification_report

#pip install livelossplot
#from livelossplot import PlotLossesKeras

# Parametri
nb_classes = 2

img_width = 224
img_height = 224


batch_size = 2
nb_epochs = 50
patience = 50
drop = 0.2

base_dir = '/home/bucci/Tirocinio/dataset/TASK_24/Fold1/'

#train_dir = base_dir + "/Train/"
#validation_dir = base_dir + "/Val/"
#test_dir = base_dir + "/Test/"

enableFinetuning = True


#------------------------------------------------------------------------------
#                           DEFINING FUNCTIONS
#------------------------------------------------------------------------------


def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt
  


def get_nb_plot(directory):
  if not os.path.exists(directory):
    print("path not found")
    return 0
  cnt_tl = 0
  cnt_ft = 0
  cnt_tl_weight=0
  cnt_ft_weight=0
  for files in glob.glob(directory):
      cnt_tl += len(glob.glob(files + '/tl_*.png'))
      cnt_ft += len(glob.glob(files + '/ft_*.png'))
      cnt_tl_weight += len(glob.glob(files + '/*_tl.h5'))
      cnt_ft_weight += len(glob.glob(files + '/*_ft.h5'))
  return cnt_tl, cnt_ft, cnt_tl_weight, cnt_ft_weight 
  


def plot_training(hist, save_dir, task, fold):

    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.figure(figsize=(80, 80))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy', linewidth=3.0)
    plt.plot(val_acc, label='Validation Accuracy', linewidth=3.0)
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy', fontsize=50)
    # plt.ylim([min(plt.ylim()),1])
    plt.title(task+' '+fold+ '\nTraining and Validation Accuracy',fontsize=50)

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss',linewidth=3.0)
    plt.plot(val_loss, label='Validation Loss',linewidth=3.0)
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy',fontsize=50)
    # plt.ylim([0,1.0])
    plt.title(task+' '+fold+ '\nTraining and Validation Loss',fontsize=50)
    plt.xlabel('epoch',fontsize=50)
    plt.savefig(save_dir)
    #plt.show()
    
    
def evalAccuracy(confusionMatrix):
    accuracy = 0
    accuracyByRow = []
    tot = 0
    for i in range(len(confusionMatrix)):
        totRow = 0
        for j in range(len(confusionMatrix)):
           tot = tot + confusionMatrix[i][j]
           totRow = totRow + confusionMatrix[i][j]
        accuracy = accuracy + confusionMatrix[i][i]
        accuracyByRow.append((confusionMatrix[i][i] * 100.0)/totRow)

    accuracy = (100.0 * accuracy)/tot

    return accuracy, accuracyByRow
    



def train(args):

    base_dir = args.base_dir
    train_dir = base_dir + "Train/"
    validation_dir = base_dir + "Val/"
    
    nb_epochs = int(args.nb_epoch)
    batch_size = int(args.batch_size)
    patience = int(args.patience)
    modelToUse = args.base_model
    drop = float(args.dropout)
    weights = args.weights
    save_fold = args.save_fold
    lr = 1e-3
    
    # -------------------------------------------------------------------------------
    #                             Build Model
    # -------------------------------------------------------------------------------

    base_model = EfficientNetB0(weights=weights, include_top=False, input_shape=(img_width,img_height,3))
    print("\nModel " + modelToUse + " selected\n")
    
    # Freeze the pretrained weights
    base_model.trainable = False

    # Rebuild top
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(drop)(x)
    predictions = layers.Dense(nb_classes, activation="softmax")(x)

    # COMPILE
    model = Model(base_model.input, predictions)
    #model.summary()
    
    #-------------------------------------------------------------------------------
    #                                AUGUMENTATION
    #-------------------------------------------------------------------------------

    #TASK_24
    # augmentation configuration for train
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        #horizontal_flip=True,
        #vertical_flip= True,
        #fill_mode='nearest'
        )

    datagen = ImageDataGenerator(
        rescale=1. / 255
        )
        
    print("INPUT PARAMETERS: ")
    print("weights: " + weights )
    print("base_dir: " + base_dir)
    print("modelToUse: " + modelToUse)
    print("base model trainable: " + str(base_model.trainable))
    print("nb_epochs: " + str(nb_epochs))
    print("batch_size: " + str(batch_size))
    print("patience: " + str(patience))
    print("dropout: " + str(drop))
    print("Optimizer: Adam")
    print("Learning Rate = "+ str(lr))
    print("Fine Tuning:"+ str(enableFinetuning))
    print("******************************************************")
        
    print("\nAUGMENTATION: ")
    print("Rotation: 30° ")
    print("width_shift_range = 0.1")
    print("height_shift_range = 0.1")
    print("shear_range = 0.1")
    print("zoom_range = 0.1")
    #print("fill_mode = nearest")
    print("******************************************************")
    
    
    #----------------------------------------
    # inserire la Train Dir attraverso args
    #-----------------------------------------
    
    
    print("\nTrain Directory: " + train_dir)
    train_generator = train_datagen.flow_from_directory(
        directory = train_dir,
        target_size = (img_width,img_height),
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = True
        )

    print("\nValidation Directory: " + validation_dir)
    val_generator = datagen.flow_from_directory(
        directory = validation_dir,
        target_size = (img_width, img_height),
        batch_size = 1,
        class_mode = "categorical",
        shuffle = False
        )

    #-------------------------------------------------------------------------------
    #                         TRANSFERN LEARNING
    #-------------------------------------------------------------------------------

    #opt_sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True) #nesterov=True
    opt_adam = keras.optimizers.Adam(learning_rate=lr)

    loss_c="categorical_crossentropy"

    model.compile(optimizer=opt_adam, loss=loss_c, metrics=["accuracy"])
    
    task = base_dir[-14:-7]
    Fold = base_dir[-6:-1]
    
        
    # setting the folder and save name
    save_dir = 'Result/' + modelToUse + '/FT_TEST/' + base_dir[-14:] + save_fold
    num = get_nb_plot(save_dir)
    name_tl = 'tl_#'+str(num[0]+1) + '_Epoch_'+str(nb_epochs) + 'Batch'+str(batch_size) + 'DropOut_'+str(drop) + '_accuracy.png'
    figure_tl = os.path.join(save_dir, name_tl)
    
    
    train_best_weight = save_dir + str(num[2]+1)+ '_' + Fold + '_tl.h5'


    # Callbacks
    checkpoint = ModelCheckpoint(train_best_weight, monitor='val_accuracy' , save_best_only=True, verbose=1)
    stop = EarlyStopping(monitor="val_loss", patience=patience, verbose=1)
    #reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=3, min_lr=0.0001, verbose=1)
    #tensorboard = TensorBoard(log_dir="logs_transfer/{}".format(time()))



    #---------------------------------------------------------------------------------
    #                           TRAINING
    #---------------------------------------------------------------------------------
        
    hist_tl = model.fit(
        train_generator,
        epochs=nb_epochs,
        validation_data=val_generator,
        batch_size=batch_size,
        callbacks=[checkpoint]
    )
    
    plot_training(hist_tl, figure_tl, task, Fold)
    
    
    #return model
    
    print("\n*********************************************************************************************")
    print("*********************************************************************************************")
    print("                                END TRANSFERN LEARNING                                       ")
    print("*********************************************************************************************")
    print("*********************************************************************************************\n")
       
    
    #----------------------------------------------------------------------------------------
    #                                          FINE TUNING
    #----------------------------------------------------------------------------------------    
    
    if enableFinetuning:
        
        print("\n*********************************************************************************************")
        print("*********************************************************************************************")
        print("                                      FINE TUNING                                            ")
        print("*********************************************************************************************")
        print("*********************************************************************************************\n")
        
        # load best training weights
        model.load_weights(train_best_weight)
        
        
        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
        frozen = 20
        for layer in model.layers[:-frozen]:
          if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
        
        lr_ft = 1e-4
        
        opt_adam_ft = keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=opt_adam_ft, loss="categorical_crossentropy", metrics=["accuracy"])
    
        print("\nFINE TUNING PARAMETERS:")
        print("Frozen Layers: " +  str(frozen))
        print("\nLearning Rate: "+ str(lr_ft))        
        
        # setting the folder and save name
        finetuning_best_weight = save_dir + str(num[3]+1)+ '_' + Fold + '_ft.h5'        
        name_ft = 'ft_#' + str(num[1] + 1) + '_Epoch_'+str(nb_epochs) + 'Batch'+str(batch_size) + 'DropOut_'+str(drop) + 'frozen_Layers'+str(frozen) + '_accuracy.png'
        figure_ft = os.path.join(save_dir, name_ft)
        
        # callbacks
        checkpoint_ft = ModelCheckpoint(finetuning_best_weight, monitor='val_accuracy' , save_best_only=True, verbose=1)
        
        
        #---------------------------------------------------------------------------------
        #                           TRAINING
        #---------------------------------------------------------------------------------
            
        hist_ft = model.fit(
            train_generator,
            epochs=nb_epochs,
            validation_data=val_generator,
            batch_size=batch_size,
            callbacks=[checkpoint_ft]
        )
        
        print("porco dio prima del plot")
        plot_training(hist_ft, figure_ft, task, Fold)
        print("porco dio dopo il plot")
        
        
    #if not enableFinetuning:
    #    finetuning_best_weight = train_best_weight
        
    return model, train_best_weight, finetuning_best_weight
    
      
def test(args, model, best_weight):
    
    print("\n*********************************************************************************************")
    print("*********************************************************************************************")
    print("                                TESTING"                                                )
    print("*********************************************************************************************")
    print("*********************************************************************************************")

    target_names = ["HC", "PT"]
    
    base_dir = args.base_dir
    test_data_dir = base_dir + "/Test"   

    nb_test_samples = get_nb_files(test_data_dir)
    #nb_classes = len(glob.glob(test_data_dir + "/*"))
    
    
    datagen = ImageDataGenerator(
        rescale=1. / 255
        )
    
    print("Test Directory: " + test_data_dir)
    
    test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size = (img_height, img_width),
        batch_size = 1,
        shuffle = False,
        class_mode = 'categorical'        
    )
    
    print("\nLoad weights from: " + best_weight)
    model.load_weights(best_weight) 

    print("\nPrediction of " + str(nb_test_samples) + " rows.")
    #probabilities = model.predict_generator(test_generator, nb_test_samples)
    probabilities = model.predict(test_generator, nb_test_samples)

    probabilities = np.argmax(probabilities, axis=1)

    print('Confusion Matrix')
    cm = confusion_matrix(test_generator.classes, probabilities)
    print(cm)

    accuracy, accuracyByRow = evalAccuracy(cm)
    print("Accuracy: " + str(accuracy))
    print("Accuracy by row:")

    for i in range(0, len(accuracyByRow)):
        print("\t" + target_names[i] + " => " + str(accuracyByRow[i]))
    print('Classification Report')

    print(classification_report(test_generator.classes, probabilities, target_names=target_names))



if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--base_dir", default=base_dir)
    a.add_argument("--nb_epoch", default=nb_epochs)
    a.add_argument("--batch_size", default=batch_size)
    a.add_argument("--patience", default=patience)
    a.add_argument("--base_model", default="EfficientNetB0")
    a.add_argument("--dropout", default=drop)
    a.add_argument("--weights", default="imagenet")
    a.add_argument("--save_fold", default="")
    a.add_argument("--plot", action="store_true")
    args = a.parse_args()

    model, tl_weight, ft_weight = train(args)

    test(args, model, tl_weight)
    test(args, model, ft_weight)

