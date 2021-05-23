import pickle
import streamlit as st

model=pickle.load(open("spam.pkl","rb"))
vector=pickle.load(open("vectorize.pkl","rb"))

def main():
	st.title("Email Spam Detector!! 📩")
	st.write("")
	st.write("Enter a Mail to check whether Spam or not: ")
	msg=st.text_input("")
	if st.button("Predict"):
		if(msg==""):
			st.error("Enter a valid mail.Field cannot be empty")
			return
		data=[msg]
		vect=vector.transform(data).toarray()
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
