# ======================================================================
# == 0. PARAMETERS =====================================================
# ======================================================================


directory_path = './data'
output_folder = "19_10_03script"

# IN THE FUTURE, ALL THE CLASSIFIER PARAMETERS WILL GO HERE AS WELL



# ======================================================================
# == 1. IMPORTS ========================================================
# ======================================================================



# ------ 1.1. General Imports ------------------------------------------

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split #sklearn...
                                                     #.cross_validation 
                                                     # is deprecated

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from os import listdir
from os.path import isfile, join



# ------ 1.2. Specialized Imports --------------------------------------



# - - - - -  1.2.1. kNN imports  - - - - - - - - - - - - - - - - - - - -

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# - - - - -  1.2.2. RBF SVM imports  - - - - - - - - - - - - - - - - - -

from sklearn.svm import SVC

# - - - - -  1.2.3. Gaussian process imports - - - - - - - - - - - - - -



from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# - - - - -  1.2.3. Gaussian process imports - - - - - - - - - - - - - -

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# - - - - -  1.2.4. Random forest imports  - - - - - - - - - - - - - - - 

from sklearn.ensemble import RandomForestClassifier



# ======================================================================
# == 2. FUNCTION DEFINITIONS ===========================================
# ======================================================================



# ------ 2.1. High Level Functions -------------------------------------



# - - - - -  2.1.1 kNN Classification and Plotting - - - - - - - - - - -

def run_kNN(X_features, Y_labels, outfilename_base,
            brem = "", npks = "", nplt = "", dataStyle = ""):

    # X_features       -->
    # Y_labels         --> 
    # outfilename_base --> "../classificationImgs/<CURR_DATE>/"
    # brem             --> "N" or "Y"
    # npks             --> str(number of pks)
    # nplt             --> str(number of plateaus)
    # dataStyle        --> "preprocessed", "SFFS", "MDS" or "corrected"
    
    # output formatting
    
    sans = False
    if len(npks) < 1 :
    
        sans = True
    
    
    cm_title_str = ""
    outfilename  = ""
    
    class_names = np.array(['1e','b2b'])
    
    if not sans : 
    
        cm_title_str = "kNN, " + brem + npks + nplt + ",  "        \
                     + dataStyle + "\n" + str(X_features.shape[0]) \
                     + " b2b or 1e events"
                     
        outfilename  = outfilename_base + brem + npks + nplt + "_" \
                     + dataStyle + "_kNN.png"
        
    else :
    
        cm_title_str = "kNN, sans,  " + dataStyle + "\n"              \
                     + str(X_features.shape[0]) + " b2b or 1e events"
                     
        outfilename  = outfilename_base + "sans_" + dataStyle         \
                     + "_kNN.png"
    
    
    # procedure
    
    x_train, x_test, \
    y_train, y_test  = train_test_split(X_features, 
                                        Y_labels, 
                                        test_size = 0.3)
    
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    label_pred = knn.predict(x_test)
    
    # generating output
    
    np.set_printoptions(precision = 2)
    plot_confusion_matrix(y_test, label_pred, classes = class_names, 
                          normalize = True, title = cm_title_str)
    
    plt.savefig(outfilename)
    plt.close()



# - - - - -  2.1.2. RBF SVM Classification and Plotting  - - - - - - - - 

def run_RBF_SVM(X_features, Y_labels, outfilename_base,
                brem = "", npks = "", nplt = "", dataStyle = ""):

    # X_features       -->
    # Y_labels         --> 
    # outfilename_base --> "../classificationImgs/<CURR_DATE>/"
    # brem             --> "N" or "Y"
    # npks             --> str(number of pks)
    # nplt             --> str(number of plateaus)
    # dataStyle        --> "preprocessed", "SFFS", "MDS" or "corrected"
    
    # output formatting
    
    sans = False
    if len(npks) < 1 :
    
        sans = True
    
    
    cm_title_str = ""
    outfilename  = ""
    
    class_names = np.array(['1e','b2b'])
    
    if not sans : 
    
        cm_title_str = "RBF_SVM, " + brem + npks + nplt + ",  "    \
                     + dataStyle + "\n" + str(X_features.shape[0]) \
                     + " b2b or 1e events"
        
        outfilename  = outfilename_base + brem + npks + nplt + "_" \
                     + dataStyle + "_RBF_SVM.png"
        
    else :
    
        cm_title_str = "RBF_SVM, sans,  " + dataStyle + "\n"          \
                     + str(X_features.shape[0]) + " b2b or 1e events"
        
        outfilename  = outfilename_base + "sans" + dataStyle          \
                     + "_RBF_SVM.png"
    
    
    # procedure
    
    x_train, x_test, \
    y_train, y_test  = train_test_split(X_features, 
                                        Y_labels, 
                                        test_size = 0.3)
    
    rbf_svm = SVC(gamma = 'auto')
    rbf_svm.fit(x_train, y_train)
    label_pred = rbf_svm.predict(x_test)
    
    # generating output
    
    np.set_printoptions(precision=2)
    plot_confusion_matrix(y_test, label_pred, classes = class_names, 
                          normalize = True, title = cm_title_str)
    
    plt.savefig(outfilename)
    plt.close()



# - - - - -  2.1.3. Gaussian Process Classification and Plotting - - - -

def run_gaus_proc(X_features, Y_labels, outfilename_base,
                  brem = "", npks = "", nplt = "", dataStyle = ""):
    
    # X_features       -->
    # Y_labels         --> 
    # outfilename_base --> "../classificationImgs/<CURR_DATE>/"
    # brem             --> "N" or "Y"
    # npks             --> str(number of pks)
    # nplt             --> str(number of plateaus)
    # dataStyle        --> "preprocessed", "SFFS", "MDS" or "corrected"
    
    # output formatting
    
    sans = False
    if len(npks) < 1 :
    
        sans = True
    
    
    cm_title_str = ""
    outfilename  = ""
    
    class_names = np.array(['1e','b2b'])
    
    if not sans : 
    
        cm_title_str = "gausProc, " + brem + npks + nplt + ",  "    \
                     + dataStyle + "\n" + str(X_features.shape[0])  \
                     + " b2b or 1e events"
                     
        outfilename  = outfilename_base  + brem + npks + nplt + "_" \
                     + dataStyle + "_gausProc.png"
        
    else :
    
        cm_title_str = "gausProc, sans,  " + dataStyle + "\n"         \
                     + str(X_features.shape[0]) + " b2b or 1e events"
        
        outfilename  = outfilename_base + "sans_" + dataStyle         \
                     + "_gausProc.png"
    
    
    # procedure
    
    x_train, x_test, \
    y_train, y_test  = train_test_split(X_features, 
                                        Y_labels, 
                                        test_size = 0.3)
    
    gpc = GaussianProcessClassifier(1.0 * RBF(1.0))
    gpc.fit(x_train, y_train)
    label_pred = gpc.predict(x_test)
    
    # generating output
    
    np.set_printoptions(precision = 2)
    plot_confusion_matrix(y_test, label_pred, classes = class_names, 
                          normalize = True, title = cm_title_str)
    
    plt.savefig(outfilename)
    plt.close()



# - - - - -  2.1.4. Random Forest Classification and Plotting  - - - - -

def run_rand_frst(X_features, Y_labels, outfilename_base,
                  brem = "", npks = "", nplt = "", dataStyle = ""):

    # X_features       -->
    # Y_labels         --> 
    # outfilename_base --> "../classificationImgs/<CURR_DATE>/"
    # brem             --> "N" or "Y"
    # npks             --> str(number of pks)
    # nplt             --> str(number of plateaus)
    # dataStyle        --> "preprocessed", "SFFS", "MDS" or "corrected"
    
    # output formatting
    
    sans = False
    if len(npks) < 1 :
    
        sans = True
    
    
    cm_title_str = ""
    outfilename  = ""
    
    class_names = np.array(['1e','b2b'])
    
    if not sans : 
    
        cm_title_str = "randFrst, " + brem + npks + nplt + ",  "   \
                     + dataStyle + "\n" + str(X_features.shape[0]) \
                     + " b2b or 1e events"
        
        outfilename  = outfilename_base + brem + npks + nplt + "_" \
                     + dataStyle + "_randFrst.png"
        
    else :
    
        cm_title_str = "randFrst, sans,  " + dataStyle + "\n"         \
                     + str(X_features.shape[0]) + " b2b or 1e events"
        
        outfilename = outfilename_base + "sans_" + dataStyle          \
                    + "_randFrst.png"
    
    
    # procedure
    
    x_train, x_test, \
    y_train, y_test  = train_test_split(X_features, 
                                        Y_labels, 
                                        test_size = 0.3)
    
    rand_frst = RandomForestClassifier(n_estimators = 100)
    rand_frst.fit(x_train, y_train)
    label_pred = rand_frst.predict(x_test)
    
    # generating output
    
    np.set_printoptions(precision = 2)
    plot_confusion_matrix(y_test, label_pred, classes = class_names, 
                          normalize = True, title = cm_title_str)
    
    plt.savefig(outfilename)
    plt.close()



# ------ 2.2 Mid Level Functions ---------------------------------------



# - - - - -  2.2.1. Confusion Matrix Plot  - - - - - - - - - - - - - - -

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize = False,
                          title = None,
                          cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if not title:
    
        if normalize:
        
            title = 'Normalized confusion matrix'
            
        else:
        
            title = 'Confusion matrix, without normalization'
            
            

    # Compute confusion matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    
    classes = classes[unique_labels(y_true, y_pred)]
    
    if normalize:
        
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
        
    else:
        
        print('Confusion matrix, without normalization')

    with open( ('log' + output_folder + '.txt'), 'a') as logfile :
    
        logfile.write( np.array_str(cm) )
    
        
    print(cm)

    fig, ax = plt.subplots( figsize = (4,4) )
    im = ax.imshow(cm, interpolation = 'nearest', cmap = cmap)
    ax.figure.colorbar(im, ax = ax)
    
    # We want to show all ticks...
    
    ax.set(xticks = np.arange(cm.shape[1]),
           yticks = np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels = classes, yticklabels = classes,
           title = title,
           ylabel = 'True label',
           xlabel = 'Predicted label')

    # Rotate the tick labels and set their alignment.
    
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right",
             rotation_mode = "anchor")

    # Loop over data dimensions and create text annotations.
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i in range(cm.shape[0]):
    
        for j in range(cm.shape[1]):
        
            ax.text(j, i, format(cm[i, j], fmt),
                    ha = "center", va = "center",
                    color = "white" if cm[i, j] > thresh else "black")
                    
                                      
    fig.tight_layout()
    ax.set_ylim(1.5,-0.5)
    
    return ax



# ======================================================================
# == 3. RUNNING SCRIPT =================================================
# ======================================================================



# ------ 3.1. Preparing Log File ---------------------------------------



logfile_cleanup = open('log' + output_folder + '.txt', 'w')
logfile_cleanup.write(' \n') 
logfile_cleanup.close()



# ------ 3.2. Importing Files from Data Folder -------------------------



files = [ f for f in listdir(directory_path)
            if isfile( join(directory_path, f) ) ]

files_b2b = sorted( [f for f in files if f[:3] == 'b2b'] )
files_one = sorted( [f for f in files if f[:3] == 'one'] )

files_both_list = [files_b2b, files_one]

files_both_DF = pd.DataFrame(files_both_list)
files_both_DF = files_both_DF.transpose()
files_both_DF = (directory_path + '/') + files_both_DF



# ------ 3.3. Performing All Classifications on All Imported Files -----



plt.ioff()

for i in range(files_both_DF.shape[0]) :
    
    b2b_D = pd.read_excel(files_both_DF.at[i,0])
    one_D = pd.read_excel(files_both_DF.at[i,1])
    
    brem = ""
    npks = ""
    nplt = ""
    
    if len(files_both_DF.at[i,0].split('_')) >= 5 : # has categorization
    
        brem = files_both_DF.at[i,0].split('_')[1]
        npks = files_both_DF.at[i,0].split('_')[2]
        nplt = files_both_DF.at[i,0].split('_')[3]
    
    
    dataStyle = files_both_DF.at[i,0].split('_')[-1].split('.')[0] 
    #    "SFFS", "MDS", etc.
    
    # UNCOMMENT THIS IF YOU WANT EQUAL b2b / 1e STATISTICS
    #b2b_D = b2b_D[0:min(b2b_D.shape[0],one_D.shape[0])]
    #one_D = one_D[0:min(b2b_D.shape[0],one_D.shape[0])]
    #    UNCOMMENT THIS IF YOU WANT EQUAL b2b / 1e STATISTICS  
    
    X_features_D = pd.concat([b2b_D, one_D], ignore_index = True)
    
    X_features = X_features_D.to_numpy()
    
    b2b_labels = np.ones( (b2b_D.shape[0],), dtype=int)# 1 == found 0vbb
    one_labels = np.zeros((one_D.shape[0],), dtype=int)# 0 == found 1e
    Y_labels = np.concatenate((b2b_labels,one_labels), axis=None)
    
    outfilename_base = "./classificationImgs/" + output_folder + "/"
    
    
    with open( ('log' + output_folder + '.txt'), 'a') as logfile :

        print("----------------------------------" \
            + "----------------------------------")
        print(" ")
        print(files_both_DF.at[i,0])
        print(" ")
        print("    running kNN classification")
        print(" ")
        
        logfile.write("----------------------------------" \
                    + "----------------------------------\n")
        logfile.write(" \n")
        logfile.write(files_both_DF.at[i,0])
        logfile.write(" \n")
        logfile.write("    running kNN classification\n")
        logfile.write(" \n")

    
    run_kNN(X_features, Y_labels, outfilename_base, 
            brem, npks, nplt, dataStyle)
    
    with open( ('log' + output_folder + '.txt'), 'a') as logfile :
    
        print(" ")
        print("    running RBF SVM classification")
        print(" ")
        
        logfile.write(" \n")
        logfile.write("    running RBF SVM classification\n")
        logfile.write(" \n")
    
    
    run_RBF_SVM(X_features, Y_labels, outfilename_base, 
                brem, npks, nplt, dataStyle)
    
    with open( ('log' + output_folder + '.txt'), 'a') as logfile :
    
        print(" ")
        print("    running gaussian process classification")
        print(" ")
        
        logfile.write(" \n")
        logfile.write("    running gaussian process classification\n")
        logfile.write(" \n")
    
    
    run_gaus_proc(X_features, Y_labels, outfilename_base, 
                  brem, npks, nplt, dataStyle)
    
    with open( ('log' + output_folder + '.txt'), 'a') as logfile :
    
        print(" ")
        print("    running random forest classification")
        print(" ")
        
        logfile.write(" \n")
        logfile.write("    running random forest classification\n")
        logfile.write(" \n")
    
    
    run_rand_frst(X_features, Y_labels, outfilename_base, 
                  brem, npks, nplt, dataStyle)
    
    with open( ('log' + output_folder + '.txt'), 'a') as logfile :
    
        print(" ")
        print(" ")
        print(" ")
        
        logfile.write(" \n")
        logfile.write(" \n")
        logfile.write(" \n")
    
    
print("script finished running.")
