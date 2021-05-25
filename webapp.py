from nltk.tokenize import RegexpTokenizer,word_tokenize
from nltk.stem import PorterStemmer
import pickle
import streamlit as st

model=pickle.load(open("spam.pkl","rb"))
vector=pickle.load(open("vectorize.pkl","rb"))

tokenizer=RegexpTokenizer('\w+')
sw=['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
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
