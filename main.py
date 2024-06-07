import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
def clean_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)


bow = pickle.load(open('bow.pkl', 'rb'))
model = pickle.load(open('nbmodel.pkl', 'rb'))

st.title('News Classifier')
st.subheader('Enter the News')
input_news = st.text_area("")
processed_news = clean_text(input_news)

vector_input = bow.transform([processed_news])

result = model.predict(vector_input)[0]

if st.button('Predict'):
    if result == 0:
        st.header('Business')
    elif result == 1:
        st.header('Entertainment')
    elif result == 2:
        st.header('Politics')
    elif result == 3:
        st.header('Sports')
    else:
        st.header('Technology')