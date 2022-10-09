import pickle
import json
from flask import Flask
from flask import request
from flask import jsonify

input_file='model2.bin'
with open(input_file,'rb') as f_in:
    model = pickle.load(f_in)

input_file2='dv.bin'
with open(input_file2,'rb') as f_in:
    dv = pickle.load(f_in)

app =Flask('predict')
@app.route('/predict', methods=['POST'])

def predict():
    test_cust = request.get_json()
    X = dv.transform([test_cust])
    predict = model.predict_proba(X)[0,1]

    result = {
            'cc_emission_probability': float(predict)
    }
    return jsonify(result)

def predict_from_file():
    with open('cust.json') as json_file:
        test_cust = json.load(json_file)
        X= dv.transform([test_cust])
        predict = model.predict_proba(X)[0,1]
        #stringret = str(predict)
        stringret = "assoreta"
    return stringret


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)


