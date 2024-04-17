from flask import Flask,request,render_template,jsonify
import os
from pyspark.sql import SparkSession

from domain.pipeline.prediction_pipeline import CustomData,PredictPipeline
application = Flask(__name__)

app = application
spark = SparkSession.builder.master('local').appName('sparkProject').getOrCreate()
## Route for a home page

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict()
        print(data)
        custom_data = CustomData(
            age=data.get('age'),
            bmi=float(data.get('bmi')),
            children=data.get('children'),
            sex=data.get('sex'),
            smoker=data.get('smoker'),
            region=data.get('region')
        )
        pred_df = custom_data.get_data_as_data_frame(spark=spark)
        print(pred_df)
        
        predict_pipeline = PredictPipeline(
            preprocessor_path = os.path.join('artifacts','preprocessor_rf'),
            model_path = os.path.join('artifacts','model_rf')
            )
        results = predict_pipeline.predict(pred_df)
        # return render_template('home.html',results=results[0])
        prediction = {"Predicted Charges": results[0]}
        return jsonify(prediction)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)