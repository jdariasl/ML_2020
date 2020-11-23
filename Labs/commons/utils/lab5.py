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
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, SVC
import warnings
import os
import itertools
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')


def generate_data(is_class = False, cols = 2):
    yy = np.random.choice(2, 30) if is_class else 2*np.random.rand(60).reshape((int(60/cols),cols))
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


def predict_svr(x_train, y_train, x_test, kernel, gamma, param_reg):
    params = {'kernel': kernel, 'gamma': gamma, 'C': param_reg}
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print("*** calculando predicciones ***")
    md = SVR(**params).fit(X_train,y_train)
    ypred= md.predict(x_test)
    return(ypred)


@unknow_error
def test_clean_data(func):
    db = np.loadtxt('AirQuality.data',delimiter='\t')  # Assuming tab-delimiter
    db = db.reshape(9357,13)
    to_remove = -200*np.ones(( 1, db.shape[1]))
    to_impute = np.hstack([to_remove[:,0:12], np.array([[10]])])
    db = np.vstack((db[0:3,:], to_remove, to_impute))
    xx, yy = func(db)
    tests = {'No se están removiendo valores faltantes en variable de respuesta': yy.shape[0] == db.shape[0] - 1,
             'no se estan imputando los valores': ut.are_np_equal(np.round(np.mean(xx, axis = 0)), np.round(xx[-1])),
             'no se estan removiendo todos los valores faltantes': not((xx==-200).any())
             }

    test_res = ut.test_conditions_and_methods(tests)

    return (test_res)


@unknow_error
def experiementarSVR(func):
    xx, yy = generate_data(False, cols=1)
    ks = ['linear','rbf']
    gs = [1.0, 0.1]
    cs = [100]
    cols= ['kernel', 'gamma', 'param_reg', 'error de prueba (promedio)',
       'error de prueba (intervalo de confianza)', '% de vectores de soporte']

    cols_errs =['error de prueba (promedio)']

    res, df_r = ut.test_experimento_oneset(func,  
                                    shape_val=(len(ks)*len(gs)*len(cs), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    x = xx,
                                    y=yy,
                                    kernels = ks,
                                    gammas= gs,
                                    params_reg = cs,
                                    return_df = True)
    
    code_to_look = ['KFold', 'kernel=kernel', 'gamma=gamma', 'C=param_reg', 'SVR',
                    'StandardScaler()', '.fit(X=X_train', 
                    '.predict(X=X_test', 
                    '.support_', 'mean_squared_error'] 
    res2 = ut.check_code(code_to_look, func, debug = False)

    cond =( (df_r['% de vectores de soporte'].max() > 100.0) or 
            (df_r['% de vectores de soporte'].min() < 0.0) or
            df_r['% de vectores de soporte'].max() <= 1.01
            )

    if (cond ):
        print("*** recordar retornar el porcentaje de vectores de soporte ***")
        return (False)

    if ( (df_r['error de prueba (intervalo de confianza)'] == df_r['error de prueba (promedio)']).all()):
        print("*** recordar retornar el intervalo de confianza ***")
        return (False)

    return (res and res2)


@unknow_error
def experiementarSVC(func):
    xx, yy = generate_data(True)
    ks = ['linear','rbf']
    gs = [1.0, 0.1]
    cs = [100]
    cols= ['kernel', 'gamma', 'param_reg', 'error de entrenamiento',
       'error de prueba', '% de vectores de soporte']
    
    cols_errs =['error de prueba', 'error de entrenamiento']

    res, df_r = ut.test_experimento_oneset(func,  
                                    shape_val=(len(ks)*len(gs)*len(cs), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    x = xx,
                                    y=yy,
                                    kernels = ks,
                                    gammas= gs,
                                    params_reg = cs,
                                    return_df = True)
    
    code_to_look = ['StratifiedKFold', 'kernel=kernel', 'gamma=gamma', 'C=param_reg', 'SVC',
                    'StandardScaler()', '.fit(X=X_train', 
                    '.predict(X=X_test',  '.predict(X=X_train',
                    '.support_', 'accuracy_score'] 
    res2 = ut.check_code(code_to_look, func, debug = False)

    cond =( (df_r['% de vectores de soporte'].max() > 100.0) or 
            (df_r['% de vectores de soporte'].min() < 0.0) or
            df_r['% de vectores de soporte'].max() <= 1.01
            )

    if (cond):
        print("*** recordar retornar el porcentaje de vectores de soporte ***")
        return (False)


    return (res and res2)



def part_2 ():
    GRADER = Grader("lab5_part2", num_questions = 4)
    db = np.loadtxt('AirQuality.data',delimiter='\t')  # Assuming tab-delimiter
    db = db.reshape(9357,13)
    db = db[0:2000, :]
    print("Dim de la base de datos original: " + str(np.shape(db)))
    GRADER.add_test("ejercicio1", Tester(test_clean_data))
    GRADER.add_test("ejercicio2", Tester(experiementarSVR))
    GRADER.add_test("ejercicio3", Tester(experiementarSVC))

    return(GRADER, db)