import nltk
import json
import nltk
import pickle
import random
from datetime import datetime
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from flask import Flask, render_template, url_for, request, jsonify
from keras.models import load_model
import requests
import requests
from bs4 import BeautifulSoup
# from bs4 import BeautifulSoup

stemmer = PorterStemmer()

model = load_model("C:\\Users\\vramt\\OneDrive\\Desktop\\chatbottt\\chatbot.h5")

all_words = []
with open("C:\\Users\\vramt\\OneDrive\\Desktop\\chatbottt\\all_words.pickle","rb") as file:
    all_words = pickle.load(file)
knn = []
with open("C:\\Users\\vramt\\OneDrive\\Desktop\\chatbottt\\dataset\\Knn.pickle","rb") as file:
    knn = pickle.load(file)


diseases_description = pd.read_csv("C:\\Users\\vramt\\OneDrive\\Desktop\\chatbottt\\dataset\\symptom_Description.csv")
diseases_description['Disease'] = diseases_description['Disease'].apply(lambda x: x.lower().strip(" "))

disease_precaution = pd.read_csv("C:\\Users\\vramt\\OneDrive\\Desktop\\chatbottt\\dataset\\symptom_precaution.csv")
disease_precaution['Disease'] = disease_precaution['Disease'].apply(lambda x: x.lower().strip(" "))
disease_precaution
symptom_severity = pd.read_csv("C:\\Users\\vramt\\OneDrive\\Desktop\\chatbottt\\dataset\\Symptom-severity.csv")
symptom_severity = symptom_severity.applymap(lambda s: s.lower().strip(" ").replace(" ", "") if type(s) == str else s)

list_of_symptoms = []

with open("C:\\Users\\vramt\\OneDrive\\Desktop\\chatbottt\\dataset\\list_of_symptoms.pickle","rb") as file:
    list_of_symptoms = pickle.load(file)

list_of_symptoms
all_words

tags = []
with open("C:\\Users\\vramt\\OneDrive\\Desktop\\chatbottt\\tags.pickle","rb") as file:
    tags = pickle.load(file)
tags

with open('C:\\Users\\vramt\\OneDrive\\Desktop\\chatbottt\\dataset\\list_of_symptoms.pickle', 'rb') as data_file:
    symptoms_list = pickle.load(data_file)

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stemmer.stem(w.lower()) for w in tokenized_sentence]
    
    bag = [0.]*len(all_words)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

def get_symptom(sentence):
    sentence = nltk.word_tokenize(sentence)
    bow = bag_of_words(sentence, all_words)
    res = model.predict(np.array([bow]))[0]
    # print(res)
    # ERROR_THRESHOLD = 0.25 # keeping some threshold value
    results = [[i,r] for i,r in enumerate(res) if r > 0.25] # Keeping only some threshold values
    results.sort(key=lambda x: x[1], reverse=True) # sorting based on some probabilities
    liss = []
    for r in results:
        liss.append({'intent' : tags[r[0]], 'probability': str(r[1])}) # returning classes and probabilities
    return liss

lis = []
def get_precautions():
    user_symptoms_list = input()
    if user_symptoms_list.lower() !="done":
        res = get_symptom(user_symptoms_list)
        res = res[0]['intent']
        lis.append(res)
        get_precautions()
    x_test = []
    for each in symptoms_list: 
        if each in lis:
            x_test.append(1)
        else: 
            x_test.append(0)

    x_test = np.asarray(x_test) 
    global disease
    disease = knn.predict(x_test.reshape(1,-1))[0]
    return disease

# res = get_precautions()
user_symptoms_list = set()
app = Flask(__name__)

@app.route('/')
def index():
    data = []
    user_symptoms_list.clear()
    file = open("New_Project\\static\\assets\\files\\ds_symptoms.txt", "r")
    all_symptoms = file.readlines()
    for s in all_symptoms:
        data.append(s.replace("'", "").replace("_", " ").replace(",\n", ""))
    data = json.dumps(data)
    return render_template('index.html', data=data)


flag = 100
lis11 = []

@app.route('/symptom', methods=['GET', 'POST'])
def predict_symptom():
    global flag
    if flag < 5:
        flag += 1
        print(120)
        print(lis11[flag])
        return jsonify(lis11[flag])

    lis11.clear()
    print("Request json:", request.json)
    sentence = request.json['sentence']
    print(sentence)
    if sentence.replace(".", "").replace("!","").lower().strip() == "done":
        if not user_symptoms_list:
            flag = 100
            response_sentences = random.choice(
                ["I can't know what disease you may have if you don't enter any symptoms :)",
                "Meddy can't know the disease if there are no symptoms...",
                "You first have to enter some symptoms!"])
        else:
            x_test = []
            for each in symptoms_list: 
                if each in user_symptoms_list:
                    x_test.append(1)
                else: 
                    x_test.append(0)

            x_test = np.asarray(x_test)   
            flag = 100
            x_test = np.asarray(x_test)            
            disease = knn.predict(x_test.reshape(1,-1))[0]
            print(disease) 
            
            sentence = 'fever'
            location = 'hyderabad'
            req = requests.get("https://www.askapollo.com/diseases/"+str("".join((sentence.lower()).split()))+"/"+str(location))
            soup = BeautifulSoup(req.content, "html.parser")
            print("".join((sentence.lower()).split()))
            severity = []

            for each in user_symptoms_list: 
                severity.append(symptom_severity.loc[symptom_severity['Symptom'] == each.lower().strip(" ").replace(" ", ""), 'weight'].iloc[0])
            precaution = disease_precaution[disease_precaution['Disease'] == disease.strip(" ").lower()]
            
            response_sentences = diseases_description.loc[diseases_description['Disease'] == disease.strip(" ").lower(), 'Description'].iloc[0]
            response_sentences += '\n\n\n\n\n\nPrecautions: ' + precaution.Precaution_1.iloc[0] + ", " + precaution.Precaution_2.iloc[0] + ", " + precaution.Precaution_3.iloc[0] + ", " + precaution.Precaution_4.iloc[0]
            response_sentences += "\n\n\n\n"
            if np.mean(severity) > 4 or np.max(severity) > 5:
                response_sentences = response_sentences + "<br><br>Considering your symptoms are severe, and Meddy isn't a real doctor, you should consider talking to one. :)"
            txt = soup.get_text()
            txt 
            lis = []
            lis = txt.split("appointment")
            severity = []

            for i in range(1,len(lis)):
                lis11.append(str(lis[i][:-5]))

            print(len(lis11))
            flag = 0
            user_symptoms_list.clear()
            severity.clear()
    else:
        lis = get_symptom(sentence)
        symptom = lis[0]['intent']
        prob = lis[0]['probability']
        print("Symptom:", symptom, ", prob:", prob)
        if float(prob) > .5:
            response_sentences = f"Hmm, I'm { prob }% sure this is " + symptom + "."
            user_symptoms_list.add(symptom)
        else:
            response_sentences = "I'm sorry, but I don't understand you."

        print("User symptoms:", user_symptoms_list)
    return jsonify(response_sentences.replace("_", " "))

app.run(debug=True)