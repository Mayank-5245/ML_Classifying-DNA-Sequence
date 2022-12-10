#Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
#Load models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler

#Reading the data
human_df = pd.read_table('human.txt')  # human data
chimp_df = pd.read_table('chimpanzee.txt') # chimpanzee data
dog_df = pd.read_table('dog.txt')   # dog data

#Plot settings
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 23
plt.rcParams['figure.titlesize'] = 26
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['figure.figsize'] = 7,4
sns.set_style('ticks')


#Defining functions
def normalized_confusion_matrix(y_test, conf_mat, model,species):
    _ , counts = np.unique(y_test,return_counts=True)
    conf_mat = conf_mat/counts
    plt.figure(figsize=(12,6))
    ax=sns.heatmap(conf_mat,fmt='.2f',annot=True,annot_kws={'size':20},lw=2, cbar=True, cbar_kws={'label':'% Class accuracy'})
    plt.title(f'Confusion Matrix ({model}-{species})',size=22)
    plt.xticks(size=20)
    plt.yticks(size=20)
    ax.figure.axes[-1].yaxis.label.set_size(20) # colorbar label
    cax = plt.gcf().axes[-1]                    # colorbar ticks
    cax.tick_params(labelsize=20)               # colorbar ticks
    plt.savefig(f'confusion-matrix-{model}-{species}.png',dpi=300)

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_train, y_train_pred)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1


#Perform EDA
print("dimension of human data: {}".format(human_df.shape))
print("dimension of chimpanzee data: {}".format(chimp_df.shape))
print("dimension of dog data: {}".format(dog_df.shape))

plt.figure(figsize=(9,6))
colors = ['indigo', 'red','green']
names=['Human','Chimpanzee','Dog']
bins= np.arange(7)+0.5
plt.hist([human_df['class'],chimp_df['class'],dog_df['class']],bins,color=colors,label=names)
plt.xlabel('Class')
plt.ylabel('Population')
plt.title('Class distribution',size=26)
plt.legend()
plt.tight_layout()
plt.savefig('Class_distribution.png',dpi=300)


#Prepare data for machine learning
def getKmers(sequence, size=6): # size=6 
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

'''
Try to convert our data sequences into short overlapping k-mers of legth 6 
Lets do that for each species of data we have using our getKmers function. 
'''

human_df['words'] = human_df.apply(lambda x: getKmers(x['sequence']), axis=1)
human_data = human_df.drop('sequence', axis=1)
chimp_df['words'] = chimp_df.apply(lambda x: getKmers(x['sequence']), axis=1)
chimp_data = chimp_df.drop('sequence', axis=1)
dog_df['words'] = dog_df.apply(lambda x: getKmers(x['sequence']), axis=1)
dog_data = dog_df.drop('sequence', axis=1)

'''
Using scikit-learn NLP to do the k-mer counting, we need to now convert the lists of k-mers for each gene into string sentences of words that the count vectorizer can use. 
We can also make a y variable to hold the class labels.
'''

human_texts = list(human_data['words'])
for item in range(len(human_texts)):
    human_texts[item] = ' '.join(human_texts[item])
y_human = human_data.iloc[:, 0].values  

'''
Same procedure for chimpanzee and dog
'''
chimp_texts = list(chimp_data['words'])
for item in range(len(chimp_texts)):
    chimp_texts[item] = ' '.join(chimp_texts[item])
y_chimp = chimp_data.iloc[:, 0].values                       
dog_texts = list(dog_data['words'])
for item in range(len(dog_texts)):
    dog_texts[item] = ' '.join(dog_texts[item])
y_dog = dog_data.iloc[:, 0].values                           


'''
Creating the Bag of Words model using CountVectorizer(), this is equivalent to k-mer counting.
The n-gram size of 4 was previously determined by testing.
'''
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(human_texts)
X_chimp = cv.transform(chimp_texts)
X_dog = cv.transform(dog_texts)

#Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_human, test_size = 0.20, random_state=42, stratify=y_human)
'''
Stratifying preserves the proportion of how data is distributed in the target column and depicts that same proportion of distribution in the train_test_split.
'''

#Machine learning models


'''
Logistic regression
'''

Clist=[1] # to find the the best value of C, Also one can use GridSearchCV from sklearn
for C in Clist : 

    logreg = LogisticRegression(C=C,solver='newton-cg').fit(X_train, y_train) #keeping C=1 a
    y_train_pred = logreg.predict(X_train)
    y_pred = logreg.predict(X_test)

    print('C : {} Training set accuracy (LOGREG) : {:.3f}'.format(C,accuracy_score(y_train, y_train_pred)))
    print('C : {} Test set accuracy (LOGREG) : {:.3f}'.format(C,accuracy_score(y_test, y_pred)))
    print('\n')

accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
conf_mat = confusion_matrix(y_test,y_pred)
normalized_confusion_matrix(y_test,conf_mat,'LOGREG','Human')
k_fold_logreg_accuracy = cross_val_score(logreg, X, y_human, cv=StratifiedKFold(n_splits=5, random_state=1, shuffle=True)) ##5-fold cross validation
k_fold_logreg_f1 = cross_val_score(logreg, X, y_human, cv=StratifiedKFold(n_splits=5, random_state=1, shuffle=True), scoring='f1_macro') ##5-fold cross validation
print(f'Average accuracy after 5 fold cross validation (LOGREG) :{k_fold_logreg_accuracy.mean().round(2)} +/- {k_fold_logreg_accuracy.std().round(2)}')
print(f'Average F1-score after 5 fold cross validation (LOGREG) :{k_fold_logreg_f1.mean().round(2)} +/- {k_fold_logreg_f1.std().round(2)}')

'''
 Predictions for the chimpanzee and dog dna sequences
'''
#Chimpanzee
y_pred_chimp=logreg.predict(X_chimp)
conf_mat = confusion_matrix(y_chimp,y_pred_chimp)
normalized_confusion_matrix(y_chimp,conf_mat,'LOGREG','CHIMPANZEE')
accuracy, precision, recall, f1 = get_metrics(y_chimp, y_pred_chimp)
print('Performace on the Chimpanzee sequence (LOGREG):')
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

#DOG
y_pred_dog=logreg.predict(X_dog)
conf_mat = confusion_matrix(y_dog,y_pred_dog)
normalized_confusion_matrix(y_dog,conf_mat,'LOGREG','DOG')
accuracy, precision, recall, f1 = get_metrics(y_dog, y_pred_dog)
print('Performace on the Dog sequence (LOGREG):')
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))


'''
Multinomial Naive Bayes
The alpha parameter was determined by grid search previously
'''

mnb = MultinomialNB(alpha=1).fit(X_train, y_train) 
y_train_pred = mnb.predict(X_train)
y_pred = mnb.predict(X_test)

print('Training set accuracy (MNB) : {:.3f}'.format(accuracy_score(y_train, y_train_pred)))
print('Test set accuracy (MNB) : {:.3f}'.format(accuracy_score(y_test, y_pred)))
print('\n')

conf_mat = confusion_matrix(y_test,y_pred)
normalized_confusion_matrix(y_test,conf_mat,'MNB','Human')
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
k_fold_mnb_accuracy = cross_val_score(mnb, X, y_human, cv=StratifiedKFold(n_splits=5, random_state=1, shuffle=True)) ##5-fold cross validation
k_fold_mnb_f1 = cross_val_score(mnb, X, y_human, cv=StratifiedKFold(n_splits=5, random_state=1, shuffle=True), scoring='f1_macro') ##5-fold cross validation
print(f'Average accuracy after 5 fold cross validation (MNB) : {k_fold_mnb_accuracy.mean().round(2)} +/- {k_fold_mnb_accuracy.std().round(2)}')
print(f'Average F1-score after 5 fold cross validation (MNB) : {k_fold_mnb_f1.mean().round(2)} +/- {k_fold_mnb_f1.std().round(2)}')

'''
 Predictions for the chimpanzee and dog dna sequences
'''
#Chimpanzee
y_pred_chimp=mnb.predict(X_chimp)
conf_mat = confusion_matrix(y_chimp,y_pred_chimp)
normalized_confusion_matrix(y_chimp,conf_mat,'MNB','CHIMPANZEE')
accuracy, precision, recall, f1 = get_metrics(y_chimp, y_pred_chimp)
print('Performace on the Chimpanzee sequence (MNB):')
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

#Dog
y_pred_dog=mnb.predict(X_dog)
conf_mat = confusion_matrix(y_dog,y_pred_dog)
normalized_confusion_matrix(y_dog,conf_mat,'MNB','DOG')
accuracy, precision, recall, f1 = get_metrics(y_dog, y_pred_dog)
print('Performace on the Dog sequence (MNB):')
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))


'''
Neural Network
'''

from sklearn.neural_network import MLPClassifier
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

mlp = MLPClassifier(random_state=0,max_iter=2000).fit(X_train_scaled, y_train)
y_pred=mlp.predict(X_test_scaled)

print('Training set accuracy (NN): {:.3f}'.format(accuracy_score(y_train, y_train_pred)))
print('Test set accuracy (NN): {:.3f}'.format(accuracy_score(y_test, y_pred)))

accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
conf_mat = confusion_matrix(y_test,y_pred)
normalized_confusion_matrix(y_test,conf_mat,'Neural Network','Human')
k_fold_mlp_accuracy = cross_val_score(mlp, scaler.fit_transform(X), y_human, cv=StratifiedKFold(n_splits=5, random_state=1, shuffle=True)) ##5-fold cross validation
k_fold_mlp_f1 = cross_val_score(mlp, scaler.fit_transform(X), y_human, cv=StratifiedKFold(n_splits=5, random_state=1, shuffle=True), scoring='f1_macro') ##5-fold cross validation
print(f'Average accuracy after 5 fold cross validation (NN) : {k_fold_mlp_accuracy.mean().round(2)} +/- {k_fold_mlp_accuracy.std().round(2)}')
print(f'Average F1-score after 5 fold cross validation (NN) : {k_fold_mlp_f1.mean().round(2)} +/- {k_fold_mlp_f1.std().round(2)}')

'''
 Predictions for the chimpanzee and dog dna sequences
'''
#Chimpanzee
y_pred_chimp=mlp.predict(X_chimp)
conf_mat = confusion_matrix(y_chimp,y_pred_chimp)
normalized_confusion_matrix(y_chimp,conf_mat,'Neural Network','CHIMPANZEE')
accuracy, precision, recall, f1 = get_metrics(y_chimp, y_pred_chimp)
print('Performace on the Chimpanzee sequence (NN):')
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

#DOG
y_pred_dog=mlp.predict(X_dog)
conf_mat = confusion_matrix(y_dog,y_pred_dog)
normalized_confusion_matrix(y_dog,conf_mat,'Neural Network','DOG')
accuracy, precision, recall, f1 = get_metrics(y_dog, y_pred_dog)
print('Performace on the Dog sequence (NN):')
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))


#Model Comparison
Accuracy = [k_fold_logreg_accuracy.mean().round(2),k_fold_mnb_accuracy.mean().round(2),k_fold_mlp_accuracy.mean().round(2)]
F1=[k_fold_logreg_f1.mean().round(2),k_fold_mnb_f1.mean().round(2),k_fold_mlp_f1.mean().round(2)]
model=['Logreg','MNB','Neural Network']
model_data=pd.DataFrame([Accuracy,F1],columns=model,index=['Accuracy','F1']).T
model_data[['Accuracy','F1']].plot.barh(figsize=(10,7))
plt.legend(frameon=False,bbox_to_anchor=(1.5,0.5), prop={'size':20})
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlim([0.65,1]);
plt.title('Model performances',size=20)
sns.despine(top=True)
plt.tight_layout()
plt.savefig('model-comparision.png',dpi=300)
