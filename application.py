import string
import pickle
import sklearn
import nltk
from nltk.corpus import stopwords
import streamlit as st

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    y = []
    x = []
    z = []

    text = text.lower()  # removing lowercase
    text = nltk.word_tokenize(text)

    for i in text:  # removing special characters
        if i.isalnum():
            y.append(i)

    for i in y:  # removing stopwords and punctuations
        if i not in stopwords.words('english') and i not in string.punctuation:
            x.append(i)

    for i in x:  # stemming
        z.append(ps.stem(i))

    text = " ".join(z)

    return text

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email / SMS Spam Classifier")
input_sms = st.text_input("Enter the message")

if st.button('Check'):
    # preprocess
    transformed_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # predict
    result = model.predict(vector_input)[0]
    # display
    if result == 1:
        print(result)
        st.header("Spam")
    else:
        print(result)
        st.header("Not Spam")


