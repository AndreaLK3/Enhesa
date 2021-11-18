import streamlit as st
import pandas as pd
import torch
import inference
from io import StringIO
import Filepaths 
import os
import measures_to_display as M
from Utils import CLASS_NAMES

def inference_on_text(text_input):
    predicted_class = inference.run_inference_on_text(text_input)
    st.session_state.class_to_display = predicted_class
    
    
def inference_on_file(uploaded_file):
    if uploaded_file is not None:
        string_io = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = string_io.read()
        inference_on_text(string_data)
    else:
        st.session_state.class_to_display = "no file uploaded"
        


# Initialization
if 'class_to_display' not in st.session_state:
    st.session_state['class_to_display'] = '-'
    st.session_state['uploaded_file'] = None
    st.session_state['processed_file'] = None
    st.session_state['vis_image']=os.path.join(Filepaths.images_folder, 'Number_of_articles.png')

    
with st.container():
    st.title("Enhesa Technical Assignment")


    st.header("Classify text")

    col1, col2  = st.columns([2,2])
    with col1:
        text_input_area = st.text_area("Predict the news category (Web, Sport, Wissenschaft etc.) of the text", value="article text", help="Insert text and click")
        st.button(label="Classify", key=1, help="", on_click=inference_on_text, kwargs={"text_input":text_input_area}) 
    with col2:
        uploaded_file = st.file_uploader(label="Upload a .txt file to use as the article text to classify", type=['.txt'])
        st.button(label="Classify", key=2, help="", on_click=inference_on_file, kwargs={"uploaded_file":uploaded_file}) 
        
            
        
    
placeholder = st.empty()
    
placeholder.text_input(label="Predicted class", value=st.session_state.class_to_display,
        help="Output of the model. To see it, insert some text above and click Classify", placeholder=None)


st.header("Model result scores")
st.text("Evaluation on test.csv. Model: CNN classifier, learning rate=1e-04")

col1, col2  = st.columns([2,2])
with col1:
        st.subheader("Confusion matrix")
        st.table(M.confusion_matrix)
with col2:
        st.subheader("Measures")
        measures_ls = list(zip(CLASS_NAMES, M.precision, M.recall, M.f1_score))
        df1 = pd.DataFrame(measures_ls, columns=["class", "precision", "recall", "f1_score"])
        st.table(df1)




st.header("Explore Dataset")

visual_radio = st.radio(label="Visualization to display", 
    options=["Number of articles per class","Vocabulary overlap between classes", 
    "Non-shared class vocabulary", "Average words in class"],
    index=0, key=None, help=None)

st.session_state['vis_image']=visual_radio

if st.session_state['vis_image'] == "Number of articles per class":
    image_fpath = os.path.join(Filepaths.images_folder, 'Number_of_articles.png')
elif st.session_state['vis_image'] == "Vocabulary overlap between classes":
    image_fpath = os.path.join(Filepaths.images_folder, 'Vocabulary_overlap.png')
elif st.session_state['vis_image'] == "Non-shared class vocabulary":
    image_fpath = os.path.join(Filepaths.images_folder, 'Vocabulary_unique.png')
elif st.session_state['vis_image'] == "Average words in class":
    image_fpath = os.path.join(Filepaths.images_folder, 'Avg_words_in_class.png')

st.image(image_fpath)

REPO_URL = ('https://github.com/AndreaLK3/Enhesa'
            'my_app/repo/')

