import os

os.chdir(".\\")
from functions import *
os.chdir(".\\Data")

data = np.load("Training_Babies.npz",allow_pickle=True)
X_train = data["T1"]
Y_train = data["Labels"]
indices = data["indices"]
X = make_patient_list(indices)[0]

#read in Thalamus Labels
data= np.load("Labels_thalamus.npz")
Labels_thal=data["Labels_thal"]
Labels_thal=make_patient_list_results(indices,Labels_thal)
L = cross_val_thalamus(Labels_thal, 5, indices)

# Create the 5 sets for crossvalidation
Y=make_labels_list(indices)[0]
X = cross_val(X,5,indices)
Y_patch = cross_val(Y,5,indices)
Y=fold_slice_labels(Y_patch)



os.chdir(".\\weights")
#train and validate on the 5 different folds, net1_crossval.hdf5 are the plic-slice selector weights
for i in range(5):
    model = PLIC_Slice_Selector()
    model.fit(X[i][1],Y[i][1],epochs=10,validation_data=( X[i][0],Y[i][0]))
    model.save_weights("net1_crossval"+str(i)+".hdf5")


# '''plot ROC curves for paper'''
liste=[]
liste2=[]
auc=[]
TPR=[]
NPV=[]
PREC=[]
SPEC=[]
FPR=[]
for i in range(5):
    model.load_weights("net1_crossval"+str(i)+".hdf5")
    predictions = model.predict(X[i][0])
    y_true = np.array(Y[i][0][:,0].flatten())
    y_scores = np.array(predictions[:,0].flatten())
    print(roc_auc_score(y_true, y_scores))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
    a = roc_auc_score(y_true,y_scores)
    thresholded = np.copy(y_scores)
    T=ROC(X[i][0],Y[i][0], predictions)
    thresholded[thresholded>T]=1
    thresholded[thresholded<1]=0
    tnr = np.sum((1-thresholded)*(1-y_true))/(np.sum((1-thresholded)))
    tpr= np.sum(thresholded*y_true)/np.sum(y_true)
    fpr = np.sum(thresholded*(1-y_true))/(np.sum(1-y_true))
    prec =  np.sum(thresholded*y_true)/( np.sum(thresholded*y_true)+np.sum(thresholded*(1-y_true)))
    spec = np.sum((1-thresholded)*(1-y_true))/(np.sum((1-thresholded)*(1-y_true))+np.sum(thresholded*(1-y_true)))
    npv = np.sum((1-thresholded)*(1-y_true))/(np.sum((1-thresholded)*(1-y_true))+np.sum((1-thresholded)*(y_true)))
    TPR.append(tpr)
    FPR.append(fpr)
    PREC.append(prec)
    SPEC.append(spec)
    NPV.append(npv)
    liste.append(fpr)
    liste2.append(tpr)
    auc.append(a)
    plot_roc_curve(fpr, tpr)
    
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.plot(liste[0], liste2[0], color='blue',alpha=0.65,linestyle="solid", label="ROC_1, AUC="+ str(np.round(auc[0],3)))
plt.plot(liste[1], liste2[1], color='red',alpha=0.65,linestyle="solid",  label="ROC_2, AUC="+ str(np.round(auc[1],3)))
plt.plot(liste[2], liste2[2], color='cyan',alpha=0.65, linestyle="solid",label="ROC_3, AUC="+ str(np.round(auc[2],3)))
plt.plot(liste[3], liste2[3], color='orange',alpha=0.65,linestyle="solid", label="ROC_4, AUC="+ str(np.round(auc[3],3)))
plt.plot(liste[4], liste2[4], color='green',alpha=0.65,linestyle="solid", label="ROC_5, AUC="+ str(np.round(auc[4],3)))  

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend()
plt.show()  
# Check if the list contains the value 

 

#loading the weights and predict for each of the 5 folds separately, here we obtain the patches used for axial segmentation (ROI-Restriction)
lst=[]
indi=[]
thal=[]
for i in range(5):
    model.load_weights("net1_crossval"+str(i)+".hdf5")
    predictions= model.predict(X[i][0])
    Xseg_test=predict_classifier(X[i][0],Y_patch[i][0],indices,predictions,ROC(X[i][0],Y[i][0], predictions))[0]
    Yseg_test=predict_classifier(X[i][0],Y_patch[i][0],indices,predictions,ROC(X[i][0],Y[i][0], predictions))[1]
    score=Yseg_test[:,0]
    ind_test=predict_classifier(X[i][0],Y_patch[i][0],indices,predictions,ROC(X[i][0],Y[i][0], predictions))[2]
    ind_test_n  = [x + np.sum(indices[:i*20]) for x in ind_test]
#####We also pick out only those Thalamus labels, that are relevant in terms of PLIC   
    Lab_thal_test = L[i][0][ind_test]
    predictions = model.predict(X[i][1])
    Xseg_train=predict_classifier(X[i][1],Y_patch[i][1],indices,predictions,ROC(X[i][1],Y[i][1], predictions))[0]
    Yseg_train=predict_classifier(X[i][1],Y_patch[i][1],indices,predictions,ROC(X[i][1],Y[i][1], predictions))[1]
    ind_train=predict_classifier(X[i][1],Y_patch[i][1],indices,predictions,ROC(X[i][1],Y[i][1], predictions))[2]
    Lab_thal_train = L[i][1][ind_train]

    indi.append([ind_test_n,ind_train])    
    lst.append([Xseg_test, Yseg_test,Xseg_train, Yseg_train, Lab_thal_train, Lab_thal_test])
    
os.chdir(".\\Data")    
#save the results obtained
np.savez_compressed("Results_class_crossval.npz", results=lst, indi = indi)
