# Enhesa Technical Assignment

## Tasks

### Create a demo application to demonstrate the results
Command: `streamlit run my_app.py`  <br/>
The demo app can be viewed in a browser, at `http://localhost:8501` <br/>
It allows one to classify a text, either via manual insertion or uploading a .txt file.<br/>
It displays the evaluation scores (F1-score, etc.) of the model used for classification on `test.csv`<br/>
Finally, it shows the visualizations made on the 10kGNAD dataset.

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
The visualizations can be accessed more easily via the Streamlit app defined in `my_app.py`, but if we wish to create 
the images anew and visualize them, it can be done with the commands: <br/>
`python`<br/>
`>>>import DatasetGraphics.ExploreDataset as ED`<br/>
`>>>ED.all_visualizations()`<br/>
(waiting time ~30 seconds)


## Model architecture
The classifier is based on a Convolutional Neural Network. <br/>
- Word embeddings of shape (vocabulary_size, 200) are randomly initialized following a normal distribution N(0,0.01). <br/>
- Having gathered the word embeddings of the article text in a matrix, 2 convolution operations are applied in parallel,
  with kernels of size 3 and 6. The dimensionality is brought from 200 to 100.
- Global max-pooling is applied, obtaining 1 vector of features from a text of arbitrary length
- The 2 feature vectors coming from (kernels + max-pooling) are concatenated, and constitute the 
input of a Linear FF-NN layer. Finally, softmax is applied for classification

## Evaluation scores
Different values of the learning rate were used in experiments, leading to different results.


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
