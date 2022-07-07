# ML_Classifying-DNA-Sequence
## Goal
Classification of a gene family based on the dna sequence.


  <img src="https://github.com/Mayank-5245/ML_Classifying-DNA-Sequence/blob/master/images/dna_sequence.jpg" align="right" width='400' height= '400' />

## Description
A genome is a complete collection of DNA in an organism. All living species possess a genome, but they differ considerably in size. A human genome has 
about 6 billion characters or letters. If you think the genome(the complete DNA sequence) is like a book, it is a book about 6 billion letters of 
“A”, “C”, “G” and “T”. In this project I have dna sequence (feature, X) of three spceies i.e. Human, Chimpanzee and Dog and each sequence belongs to certain gene family (target, y).
This is a multiclassification problem. I am using k-mers counting technique and multiple machine learning model to predict the gene family based on the DNA sequence.

  <img src="https://github.com/Mayank-5245/ML_Classifying-DNA-Sequence/blob/master/images/class.png" width="300" />

## Data Preprocessing

### k-Mer counting technique
In bioinformatics, k-mers are substrings of length k contained within a biological sequence. Primarily used within the context of computational genomics 
and sequence analysis, in which k-mers are composed of nucleotides (i.e. A, T, G, and C), k-mers are capitalized upon to assemble DNA sequences,improve
heterologous gene expression, identify species in metagenomic samples, and create attenuated vaccines.

k-mers are simply length k subsequences. For example, all the possible k-mers of a DNA sequence are shown below:

k-mers for GTAGAGCTGT

1-->G, T, A, G, A, G, C, T, G, T

2-->GT, TA, AG, GA, AG, GC, CT, TG, GT

3-->GTA, TAG, AGA, GAG, AGC, GCT, CTG, TGT 

and so on.


### EDA
#### Class Distribution

<img src="https://github.com/Mayank-5245/ML_Classifying-DNA-Sequence/blob/master/images/Class_distribution.png" >


## Machine Learning models

-  Logistic regression
- Multinomial Naive Bayes
-  Neural network


## Results
### Model Comparison

<p float="left">
  <img src="https://github.com/Mayank-5245/ML_Classifying-DNA-Sequence/blob/master/images/model-comparision.png" />
</p>


### Confusion Matrix 
<p float="left">
  <img src="https://github.com/Mayank-5245/ML_Classifying-DNA-Sequence/blob/master/images/confusion-matrix-LOGREG-Human.png" width="300" />
  <img src="https://github.com/Mayank-5245/ML_Classifying-DNA-Sequence/blob/master/images/confusion-matrix-MNB-Human.png" width="300" /> 
  <img src="https://github.com/Mayank-5245/ML_Classifying-DNA-Sequence/blob/master/images/confusion-matrix-Neural Network-Human.png" width="300" />
</p>

### Confusion Matrix when we tested on Chimpanzee dataset
<p float="left">
  <img src="https://github.com/Mayank-5245/ML_Classifying-DNA-Sequence/blob/master/images/confusion-matrix-LOGREG-CHIMPANZEE.png" width="300" />
  <img src="https://github.com/Mayank-5245/ML_Classifying-DNA-Sequence/blob/master/images/confusion-matrix-MNB-CHIMPANZEE.png" width="300" /> 
  <img src="https://github.com/Mayank-5245/ML_Classifying-DNA-Sequence/blob/master/images/confusion-matrix-Neural Network-CHIMPANZEE.png" width="300" />
</p>

### Confusion Matrix when tested on Dog dataset
<p float="left">
  <img src="https://github.com/Mayank-5245/ML_Classifying-DNA-Sequence/blob/master/images/confusion-matrix-LOGREG-DOG.png" width="300" />
  <img src="https://github.com/Mayank-5245/ML_Classifying-DNA-Sequence/blob/master/images/confusion-matrix-MNB-DOG.png" width="300" /> 
  <img src="https://github.com/Mayank-5245/ML_Classifying-DNA-Sequence/blob/master/images/confusion-matrix-Neural Network-DOG.png" width="300" />
</p>


## Conclusion
- `Multinomial Naive Bayes` algorithm seems to provide the better performance than logistic regression and neural network based on 5-fold cross validation of the dataset and also achieves a higher F1-score as well, which is better metric for model evalution.
- The model seems to perform well on human data and also does well on chimpanzee as chimpanzee and human are similar genetically. 
- The performance on dog data is not quite as good since the dogs are more divergent from human genetically than the chimpanzee.
