# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

#objawy	wiek	choroby_wsp	wzrost	leki	zdrowie

# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

	st.set_page_config(page_title="Zdrowie app")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://www.meme-arsenal.com/memes/8dee702e9d8fac157225ae9ff528bd14.jpg")

	with overview:
		st.title("Zdrowie app")

	with left:
		age_slider = st.slider("wiek", value=1, min_value=1, max_value=77)
		objawy_slider = st.slider("objawy", value=1, min_value=1, max_value=5)
		leki_slider = st.slider("leki", value=1, min_value=1, max_value=4)

	with right:
		choroby_slider = st.slider("choroby_wsp", min_value=0, max_value=5)
		wzrost_slider = st.slider("wzrost", min_value=159, max_value=200)

	data = [[objawy_slider, age_slider, choroby_slider, wzrost_slider, leki_slider]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba jest zdrowa?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
