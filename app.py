import pickle 
from flask import Flask,request,app,jsonify,render_template
import numpy as np
import pandas as pd 

app=Flask(__name__)

model=pickle.load(open("IrisModel.pkl","rb"))

@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/predict_api",methods=["POST"])
def predict_api():
    data=request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    passed_value=np.array(list(data.values())).reshape(1,-1)
    output=model.predict(passed_value)
    print(output[0])

    return jsonify(output[0]) 

@app.route("/predict",methods=["POST"])
def predict():
    data=[float(x) for x in request.form.values()]
    print(data)
    passed_value=np.array(data).reshape(1,-1)
    output=model.predict(passed_value)
    print(output[0])

    if(output[0]=="Iris-virginica"):
        predict_img="virginica.png"
    elif(output[0]=="Iris-versicolor"):
        predict_img="versicolor.png"
    elif(output[0]=="Iris-setosa"):
        predict_img="setosa.png"

    return render_template("Home.html",predict_text=f"The Predicted Species is: {output[0]}",img=predict_img)

if __name__=="__main__": 
    app.run(debug=True)
