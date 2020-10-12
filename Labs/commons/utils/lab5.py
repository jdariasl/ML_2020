"""
Este archivo es generado automaticamente.

###### NO MODIFICAR #########

# cualquier alteración del archivo
# puede generar una mala calificación o configuracion
# que puede repercutir negativamente en la 
# calificación del laboratorio!!!!!

###### NO MODIFICAR #########
"""

from imports import *
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPRegressor
import warnings
import os
warnings.filterwarnings('ignore')


def generate_data(is_class = False):
    yy = np.random.choice(2, 30) if is_class else 2*np.random.rand(60).reshape((30,2))
    xx = np.vstack([np.random.rand(15, 3), 2*np.random.rand(15, 3)])
    return (xx, yy)

@unknow_error
def test_experimentar_elman(func):
    xx, yy = generate_data(True)
    looksbacks = [1,2]
    neu = [5,10]
    cols =['lags', 'neuronas por capa', 'error de entrenamiento',
            'error de prueba']

    cols_errs = ['error de entrenamiento','error de prueba']

    res = ut.test_experimento_oneset(func,  shape_val=(len(looksbacks)*len(neu), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    data = xx,
                                    look_backs = looksbacks,
                                    hidden_neurons= neu)
    code_to_look = ['nl.net.newelm', 'init.InitRand', '[num_hidden_neurons,1]', "nl.trans.TanSig()",  
                    'nl.trans.PureLin()' , ".init", ".train(trainX", ".sim(trainX", 
                    "epochs=1000", "goal=0.001", '.sim(testX',
                    "mean_absolute_error(testY", "mean_absolute_error(trainY"] 
    res2 = ut.check_code(code_to_look, func, msg = "**** recordar usar las funciones sugeridas ***", debug = False)
    return (res and res2)


@unknow_error
def test_experimetar_mlp(func):
    xx, yy = generate_data(True)
    looksbacks = [1,2]
    neu = [5,10]
    cols =['lags', 'neuronas por capa', 'error de entrenamiento',
            'error de prueba']

    cols_errs = ['error de entrenamiento','error de prueba']

    res = ut.test_experimento_oneset(func,  shape_val=(len(looksbacks)*len(neu), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    data = xx,
                                    look_backs = looksbacks,
                                    hidden_neurons= neu)
    code_to_look = ['MLPRegressor',  'hidden_layer_sizes=(num_hidden_neurons',
                    "max_iter=1000", 'random_state=10', '.fit', '.predict',
                    "mean_absolute_error(testY", "mean_absolute_error(trainY"] 
    res2 = ut.check_code(code_to_look, func, msg = "**** recordar usar las funciones sugeridas ***", debug = False)
    return (res and res2)

    return (res and res2)

@unknow_error
def test_experimentar_LSTM(func):
    xx, yy = generate_data(True)
    looksbacks = [1,2]
    neu = [5]
    cols =['lags', 'neuronas por capa', 'error de entrenamiento',
            'error de prueba']

    cols_errs = ['error de entrenamiento','error de prueba']

    res = ut.test_experimento_oneset(func,  shape_val=(len(looksbacks)*len(neu), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    data = xx,
                                    look_backs = looksbacks,
                                    hidden_neurons= neu)
    code_to_look = ['create_tf_model',   'epochs=100', 
                    '.predict(trainX)', '.predict(testX)',
                    "mean_absolute_error(testY", "mean_absolute_error(trainY"] 
    res2 = ut.check_code(code_to_look, func, msg = "**** recordar usar las funciones sugeridas ***", debug = False)
    return (res and res2)


def part_1 ():
    GRADER = Grader("lab5_part1", num_questions = 4)
    dataset = pd.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
    dataset.columns =[ 'passengers']
    os.system("pip install neurolab")
    os.system("pip install statsmodels==0.12")
    GRADER.add_test("ejercicio1", Tester(test_experimentar_elman))
    GRADER.add_test("ejercicio2", Tester(test_experimetar_mlp))
    GRADER.add_test("ejercicio3", Tester(test_experimentar_LSTM))


    return(GRADER, dataset)