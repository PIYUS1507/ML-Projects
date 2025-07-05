from flask import Flask,request,render_template
from src.pipelines.predict_pipeline import CustomData,PredictPipeline


app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predictData',methods=['GET','POST'])
def prediction_data():
    if request.method=='GET':
        return render_template("home.html")
    
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pref_df=data.get_data_data_frame()



        model=PredictPipeline()
        result=model.predict(pref_df)

        return render_template('home.html',results=result[0])


if __name__=="__main__":
    app.run(debug=True)