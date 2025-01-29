from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__) # Entry point of flak

app=application

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html') #Under template folder we need index.html

@app.route('/predictdata',methods=['GET','POST']) #It supports 2 moethods GET and POST
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
                            
        data=CustomData(
            account = float(request.form.get('account')), 
            partition = int(request.form.get('partition')), 
            priority = float(request.form.get('priority')) ,
            timelimit = float(request.form.get('timelimit')), 
            CPU_ALLOC_TRES = float(request.form.get('CPU_ALLOC_TRES')),
            NODE_ALLOC_TRES = float(request.form.get('NODE_ALLOC_TRES')),
            BILLING_ALLOC_TRES = float(request.form.get('BILLING_ALLOC_TRES')),
            CPU_REQ_TRES = float(request.form.get('CPU_REQ_TRES')),
            MEMORY_REQ_TRES = float(request.form.get('MEMORY_REQ_TRES')),
            NODE_REQ_TRES = float(request.form.get('NODE_REQ_TRES')),
            BILLING_REQ_TRES = float(request.form.get('BILLING_REQ_TRES')) ,
            Queue_time = float(request.form.get('Queue_time'))  
                    
            
        )

        # Convert the data to a DataFrame using Predict_pipeline.py 
        predict_df = data.get_data_as_data_frame() 
        print(predict_df)

        # Load the trained model (assuming you have loaded it earlier in your code)
        predict_pipeline= PredictPipeline()

        results = predict_pipeline.predict(predict_df) ## Calling predict function adn giving dataframe
        
        # Return prediction result
        #return render_template('result.html', prediction=prediction)
       
        #return render_template('home.html', results=results[0]:,.2f)
        return render_template('home.html', results="{:,.2f}".format(results[0]))
        
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
