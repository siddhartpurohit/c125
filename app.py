from flask import Flask,jsonify,request
from classifier import get_prediction

app = Flask(__name__)

@app.route("/")
def index():
    return "Welcome to the home page"

@app.route("/predict-digit" , methods = ["POST"] )
def predict_data():
    img = request.files.get("digit")
    p = get_prediction(img)
    return jsonfiy({
            "Prediction":p 

    }),200

if(__name__ =="__main__"):
    app.run(debug=True)
    
