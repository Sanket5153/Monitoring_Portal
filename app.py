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
            id_array_job=int(request.form.get('id_array_job')),
            id_array_task=int(request.form.get('id_array_task')),
            id_user=int(request.form.get('id_user')),
            kill_requid=int(request.form.get('kill_requid')),
            nodes_alloc=int(request.form.get('nodes_alloc')),
            nodelist=request.form.get('nodelist'),
            cpus_req=int(request.form.get('cpus_req')),
            derived_ec=int(request.form.get('derived_ec')),
            exit_code=request.form.get('exit_code'),
            gres_req=request.form.get('gres_req'),
            gres_alloc=request.form.get('gres_alloc'),
            array_max_tasks=int(request.form.get('array_max_tasks')),
            array_task_pending=int(request.form.get('array_task_pending')),
            constraints=request.form.get('constraints'),
            flags=request.form.get('flags'),
            mem_req=int(request.form.get('mem_req')),
            partition=request.form.get('partition'),
            priority=int(request.form.get('priority')),
            state=request.form.get('state'),
            timelimit=int(request.form.get('timelimit')),
            time_submit=int(request.form.get('time_submit')),
            time_eligible=int(request.form.get('time_eligible')),
            time_start=int(request.form.get('time_start')),
            
            time_suspended=int(request.form.get('time_suspended')),
            track_steps=int(request.form.get('track_steps')),
            tres_alloc=request.form.get('tres_alloc'),
            tres_req=request.form.get('tres_req'),
            job_type=request.form.get('job_type'),
            id_job=int(request.form.get('id_job'))

        )

        # Convert the data to a DataFrame using Predict_pipeline.py 
        predict_df = data.get_data_as_data_frame() 
        print(predict_df)

        # Load the trained model (assuming you have loaded it earlier in your code)
        predict_pipeline= PredictPipeline()
        results = predict_pipeline.predict(predict_df) ## Calling predict function adn giving dataframe

        # Return prediction result
        #return render_template('result.html', prediction=prediction)
        return render_template('result.html', results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
