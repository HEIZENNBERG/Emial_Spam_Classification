import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


modelNavies_Bais = pickle.load(open('NaivesBaies.pkl','rb'))
modelLogistique = pickle.load(open('Logistique.pkl','rb'))
modelmlp = pickle.load(open('modelmlp.pkl','rb'))


cv=pickle.load(open('vectorizer.pkl','rb'))


def classify_email(msg, selected_model):
    data = [msg]
    vec = cv.transform(data).toarray()
    if selected_model == 'Mlp Classifier':
        model = modelmlp
    elif selected_model == 'Logistic Regression':
        model = modelLogistique
    else:
        model = modelNavies_Bais
    
    result = model.predict(vec)
    return result[0]

def main():
    st.title("Email Spam Classification")
    st.subheader("Classification")
    
    msg = st.text_input("Enter a text")
    selected_model = st.selectbox("Select Classification Algorithm", 
                                  ['Logistic Regression', 'Naive Bayes',"Mlp Classifier"])
    
    if st.button("Process"):
        if classify_email(msg, selected_model) == 0:
            st.success("This is Not A Spam Email")
        else:
            st.error("This is A Spam Email")

main()
