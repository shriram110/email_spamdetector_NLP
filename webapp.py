from nltk.tokenize import RegexpTokenizer,word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
import streamlit as st

model=pickle.load(open("spam.pkl","rb"))
vector=pickle.load(open("vectorize.pkl","rb"))

tokenizer=RegexpTokenizer('\w+')
sw=set(stopwords.words('english'))
ps=PorterStemmer()

def dopreprocess(sample):
    sample=sample.lower()
    tokens=tokenizer.tokenize(sample)
    removed_stopwords=[word for word in tokens if word not in sw]
    stemmed_words=[ps.stem(token)for token in removed_stopwords]
    preprocessed=' '.join(stemmed_words)
    return preprocessed

def preprocessed(document):
    docs=[]
    for doc in document:
        docs.append(dopreprocess(doc))
    return docs

def prepare(messages):
    d=preprocessed(messages)
    return vector.transform(d)

def main():
	st.title("Email Spam Detector!! 📩")
	st.write("")
	st.write("Enter a Mail to check whether it's Spam or not: ")
	msg=st.text_input("")
	if st.button("Predict"):
		if(msg==""):
			st.error("Enter a valid mail.Field cannot be empty")
			return
		data=[msg]
		vect=prepare(data)
		prediction=model.predict(vect)
		result=prediction[0]
		if result==1:
			st.error("OMG!! This is a Spam Mail")
		else:
			st.success("This is not a Spam Mail")

def about():
	st.title("Built with StreamLit and Python 🐍")
	st.write("This is a webapp used to check whether the mails are spam or not.")
	st.write("Created by V Shriram Bharadwaj")

radio =st.sidebar.selectbox("Select an option",["Home","About"]) 
if radio=="Home":
   main()
elif radio=="About":
   about()
