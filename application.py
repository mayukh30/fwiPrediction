from flask import Flask,request, jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regressor and standard scaler pickle
ridle_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))



@app.route('/') # home page
def index():
    return render_template('index.html') # this will try to find index.html in templates folder

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':

        # 1. get the data from the form
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        # 2. scale the new data point
        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        # 3. predict using ridge model
        result=ridle_model.predict(new_data_scaled)

        # 4. return the predicted price to the user
        return render_template('home.html', results=result[0])

    else:  ## get request
        return render_template('home.html')
    

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=4000)
    # host address = 0.0.0.0 means  it is mapped to all available IP addresses on the local machine
    # by deafult port is 5000 , you can change it by adding port=your_port_number in app.run() function