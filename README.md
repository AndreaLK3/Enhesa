# Technical Assignment

## Tasks

### Train classifier
Train a CNN-based classifier, to categorize the text of a news article.
<br/> Command: `python train_model.py` <br/>
Dataset: `train.csv` of the 10kGNAD corpus. Internally it is split into 90% training and 10% validation set <br/>
Parameters: learning_rate (float, optional)

### Evaluation and inference
Load an already-trained model. The model is either evaluated on the test set of the 10kGNAD corpus, or used for inference (classifying 1 sample that we provide)
<br/>
Command: `python inference.py --text="example article"` or `python inference.py use_test_set=True` <br/>
Dataset: either text provided by the user, or `test.csv` from the 10kGNAD corpus.<br/>
Parameters: learning_rate (float, optional), text (string), use_test_set (bool, optional)

### Exploring the 10kGNAD dataset

Four visualizations are created with `matplotlib` to investigate the 10kGNAD dataset.<br/>
The visualizations can also be accessed via the Streamlit app defined in `my_app.py`, but if we wish to create 
the images anew and visualize them, the command is: <br/>
`python draw_visuals.py`<br/>
(waiting time ~30 seconds)

### Create a demo application to demonstrate the results
Command: `streamlit run my_app.py`  <br/>
The demo app can be viewed in a browser, at `http://localhost:8501` <br/>
It allows one to classify a text, either via manual insertion or uploading a .txt file.<br/>
It displays the evaluation scores (F1-score, etc.) of the model used for classification on `test.csv`<br/>
Finally, it shows the visualizations made on the 10kGNAD dataset.



## Model architecture
The classifier is based on a Convolutional Neural Network. <br/>
- Word embeddings of shape (vocabulary_size, 200) are randomly initialized following a normal distribution N(0,0.01). <br/>
- Having gathered the word embeddings of the article text in a matrix, 2 convolution operations are applied in parallel,
  with kernels of size 3 and 6. The dimensionality is brought from 200 to 100.
- Global max-pooling is applied, obtaining a vector of features from a text of arbitrary length
- The 2 feature vectors coming from (kernels + max-pooling) are concatenated, and constitute the 
input of a Linear FF-NN layer. Finally, softmax is applied for classification
  
note: the current version of the repository also contains a model used for inference; 
file size = ~69MBs

## Evaluation scores
Different values of the learning rate were used in experiments, leading to different results.

### Scores comparison
| learning rate | NLL loss | Accuracy |  
| ------------- |----------|----------|
|       5e-4    | 0.68     | 0.806    |
|       1e-4    | 0.62     | 0.813    |
|       5e-5    | 0.68    |  0.79     |

##### F1 score 
|    learning rate      | Etat | Inland | International | Kultur | Panorama | Sport | Web | Wirtschaft | Wissenschaft |
|----------|------|--------|---------------|--------|----------|-------|-----|------------|--------------|
| 5e-4     | 0.75 | 0.76 | 0.82 | 0.78 | 0.72 | 0.96 | 0.9 | 0.75 | 0.76  |
| 1e-4     | 0.77 | 0.81 | 0.81 | 0.69 | 0.75 | 0.97 | 0.89| 0.76| 0.82 |
| 5e-5     |0.75 | 0.76 | 0.78 | 0.75 | 0.71 | 0.97  | 0.87| 0.72| 0.75|     

### Confusion matrices

#### CNN classifier, lr=5e-04
| Etat | Inland | International | Kultur | Panorama | Sport | Web | Wirtschaft | Wissenschaft |
|----|-------|----|-------|-------|-------|-------|----|----|
| 41 | 4     | 1  | 3     | 6     | 0     | 6     | 4  | 2  |
| 0  | 79    | 1  | 2     | 12    | 0     | 1     | 5  | 2  |
| 1  | 2 | 122 | 2  | 15    | 1     | 0     | 7     | 1  |    
| 1  | 0     | 2  | 45    | 2     | 0     | 1     | 0  | 3  |
| 0  | 10    | 12 | 3 | 125 | 0     | 2     | 3     | 13 |    
| 0  | 0     | 1  | 3     | 4  | 111 | 0     | 0     | 1  |   |
| 0  | 1     | 4  | 2     | 1     | 0 | 151 | 7     | 2  |    
| 0  | 9     | 3  | 2     | 14    | 0     | 6 | 102 | 5  |    
| 0  | 0     | 0  | 0     | 0     | 0     | 2     | 2  | 53 |

#### CNN classifier, lr=1e-04
| Etat | Inland | International | Kultur | Panorama | Sport | Web | Wirtschaft | Wissenschaft |
|----|-------|----|-------|-------|-------|-------|----|-----|
| 46 | 2     | 2  | 4     | 3     | 1     | 4     | 4  | 1   |
| 0  | 80    | 0  |   2   | 9     | 0     | 1     | 6  | 4  |
| 1  | 1     | 120| 1     | 17    | 1     | 0     | 10 | 0  | 
| 2  | 1     | 2  | 42    | 3     | 1     | 2     | 0  | 1  |
| 2  | 4     | 14 | 8     |128    | 0     | 0     | 9   | 3  |   
| 0  | 0     | 1  | 2     | 1     |   115 | 0     | 0     | 1  |   
| 1  | 1     | 2  | 4     | 2     | 0 | 144 | 12    | 2  |    
| 0  | 7     | 4  | 3     | 10    | 0     | 4 | 112 | 1  |   
| 1  | 0     | 0  | 2     | 2     | 0     | 2     | 1  | 49  |

#### CNN classifier, lr=1e-05

| Etat | Inland | International | Kultur | Panorama | Sport | Web | Wirtschaft | Wissenschaft |
|----|----|-----|----|-----|-----|-----|-----|----|
| 47 | 3  | 0   | 1  | 3   | 0   | 7   | 5   | 1  |
| 2  | 76 | 0   | 2  | 10  | 0   | 1   | 9   | 2  |
| 1  | 3  | 120 | 1  | 14  | 1   | 1   | 6   | 4  |
| 4  | 0  | 0   | 38 | 3   | 0   | 3   | 1   | 5  |
| 1  | 8  | 20  | 4  | 116 | 1   | 2   | 9   | 7  |
| 0  | 0  | 1   | 1  | 1   | 115 | 0   | 2   | 0  |
| 1  | 1  | 5   | 0  | 0   | 0   | 146 | 13  | 2  |
| 2  | 3  | 8   | 1  | 12  | 0   | 7   | 106 | 2  |
| 1  | 3  | 2   | 0  | 0   | 0   | 2   | 1   | 48 |








## Source code structure
- **Top-level files** <br/>
    Scripts, general utilities, StreamLit app
- Folder: **10kGNAD** <br/>
    The dataset of German news articles
- Folder: **DatasetGraphics** <br/>
    The code for the matplotlib visualizations on the dataset
  - Subfolder: **Figures** <br/>
    Contains the images
- Folder: **Model** <br/>
    The core: CNN model definition, training loop and associated utilities (vocabulary creation, corpus reader)
  - Subfolder: **Resources** <br/>
    Training and validation splits, vocabulary, word embeddings
  - Subfolder: **SavedModels**
- Folder: **MyTests** <br/>
    A few unit tests, on creating the vocabulary and loading word vectors  
