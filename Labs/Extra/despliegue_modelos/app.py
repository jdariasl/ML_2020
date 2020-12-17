  
from flask import Flask, render_template, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange
import numpy as np
from joblib import load
import os

app = Flask(__name__)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY



###
# Cargar el archivo del modelo
# verificar que es el mismo nombre
# o realizar el cambio que haya lugar
###
flower_model = load("clasificador.joblib")


###
# Funcion que factoriza 
# la prediccion
###
def return_prediction(model,sample_json):
    
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']

    flower = [[s_len,s_wid,p_len,p_wid]]

    species = model.predict(flower)

    classes = np.array(['setosa', 'versicolor', 'virginica'])


    return classes[species][0]

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

    if form.validate_on_submit():
        # Grab the data from the breed on the form.
        print(form)

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
    print(session.keys())
    content['sepal_length'] = float(session['sep_len'])
    content['sepal_width'] = float(session['sep_wid'])
    content['petal_length'] = float(session['pet_len'])
    content['petal_width'] = float(session['pet_wid'])

    results = return_prediction(model=flower_model,sample_json=content)

    return render_template('prediction.html',results=results)


if __name__ == '__main__':
    app.run(debug=True)