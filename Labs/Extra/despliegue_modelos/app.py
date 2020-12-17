  
from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange
import numpy as np
from joblib import load

app = Flask(__name__)


###
# Re factoriznado
# para obtener el etiqueta
###

# Cargando Pipeline
flower_model = load("clasificador.joblib")

def return_prediction(model,sample_json):
    
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']

    flower = [[s_len,s_wid,p_len,p_wid]]

    flower = model.predict(flower)

    classes = np.array(['setosa', 'versicolor', 'virginica'])

    class_ind = model.predict_classes(flower)

    return classes[class_ind][0]

##
# Creacion del
# Formulario
##
class FlowerForm(FlaskForm):
    sep_len = TextField('Sepal Length')
    sep_wid = TextField('Sepal Width')
    pet_len = TextField('Petal Length')
    pet_wid = TextField('Petal Width')

    submit = SubmitField('Analyze')


###
# Creacion del punto de entrada
#
###
@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = FlowerForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['sep_len'] = form.sep_len.data
        session['sep_wid'] = form.sep_wid.data
        session['pet_len'] = form.pet_len.data
        session['pet_wid'] = form.pet_wid.data

        return redirect(url_for("prediction"))


    return render_template('home.html', form=form)

###
# Creacion del punto de entrada para la prediccion
###
@app.route('/prediction')
def prediction():

    content = {}

    content['sepal_length'] = float(session['sep_len'])
    content['sepal_width'] = float(session['sep_wid'])
    content['petal_length'] = float(session['pet_len'])
    content['petal_width'] = float(session['pet_wid'])

    results = return_prediction(model=flower_model,sample_json=content)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)