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

if __name__=="__main__":
    app.run(debug=True)
