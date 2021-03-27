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
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.datasets import load_digits, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def generate_data(is_class = False, deter = False):
    yy = np.random.choice(2, 30) if is_class else 2*np.random.rand(60).reshape((30,2))
    xx = np.vstack([np.random.rand(15, 3), 2*np.random.rand(15, 3)])

    if deter:
        yy = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1])
    return (xx, yy)

@unknow_error
def test_output_activation(func):
    act = func()

    tests = {'explorar de mejor manera la libreria': act[0:2] == 'id'}
    code_to_look = ['.out_activation_']
    res2 = ut.check_code(code_to_look, func, "explorar de mejor manera la libreria")
    return (ut.test_conditions_and_methods(tests) and res2)


@unknow_error
def test_experimetar_mlp(func):
    xx, yy = generate_data()
    capas = [1,2]
    neu = [5,10]
    cols =['capas ocultas',
            'neuronas en capas ocultas',
            'error de prueba y1(media)',
            'intervalo de confianza y1',
            'error de prueba y2(media)',
            'intervalo de confianza y2']

    cols_errs = ['error de prueba y1(media)',
            'intervalo de confianza y1',
            'error de prueba y2(media)',
            'intervalo de confianza y2']

    res = ut.test_experimento_oneset(func,  shape_val=(len(capas)*len(neu), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    X = xx, Y=yy,
                                    num_hidden_layers = capas,
                                    num_neurons= neu)
    code_to_look = [['MLPRegressor', 'hidden_layer_sizes=', 'activation=', "'tanh'",  
                    'max_iter=' , ".fit", ".predict(Xtest)", "X=Xtrain,", 
                    "hidden_layers*[neurons]", "mean_absolute_percentage_error", 'multioutput=',
                    "np.mean(ErrorY1)", "np.mean(ErrorY2)"],
                    ['MLPRegressor', 'hidden_layer_sizes=', 'activation=', '"tanh"',  
                    'max_iter=' , ".fit", ".predict(Xtest)", "X=Xtrain,", 
                    "hidden_layers*[neurons]", "mean_absolute_percentage_error", 'multioutput=',
                    "np.mean(ErrorY1)", "np.mean(ErrorY2)"]]
    res2 = ut.check_code(code_to_look, func)
    return (res and res2)


@unknow_error
def test_output_activation_MPC(func):
    act = func()
    tests = {'explorar de mejor manera la libreria': act[0:2] == 'lo'}
    code_to_look = ['.out_activation_']
    res2 = ut.check_code(code_to_look, func, "explorar de mejor manera la libreria")
    return (ut.test_conditions_and_methods(tests) and res2)


@unknow_error
def test_experimetar_mlpc(func):
    xx, yy = generate_data(True, True)
    capas = [1,10]
    neu = [1,2]
    cols =['capas ocultas',
            'neuronas en capas ocultas',
            'error de prueba(media)',
            'intervalo de confianza']

    cols_errs = ['error de prueba(media)','intervalo de confianza']

    res = ut.test_experimento_oneset(func,  shape_val=(len(capas)*len(neu), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    X = xx, Y=yy,
                                    num_hidden_layers = capas,
                                    num_neurons= neu)
    code_to_look = [['MLPClassifier', 'hidden_layer_sizes=', 'activation=', "'tanh'",  
                    'max_iter=' , ".fit", ".predict(Xtest)", "X=Xtrain,",  
                     'accuracy_score', "hidden_layers*[neurons]"],
                     ['MLPClassifier', 'hidden_layer_sizes=', 'activation=', '"tanh"',  
                    'max_iter=' , ".fit", ".predict(Xtest)", "X=Xtrain,",  
                     'accuracy_score', "hidden_layers*[neurons]"]] 
    res2 = ut.check_code(code_to_look, func)
    return (res and res2)

def part_1 ():
    GRADER = Grader("lab4_part1", num_questions = 4)
    GRADER.add_test("ejercicio1", Tester(test_output_activation))
    GRADER.add_test("ejercicio2", Tester(test_experimetar_mlp))
    GRADER.add_test("ejercicio3", Tester(test_output_activation_MPC))
    GRADER.add_test("ejercicio4", Tester(test_experimetar_mlpc))
    return(GRADER)

def generate_data2():
    xx = np.array([[1,2,2], [2,2,3], [1,1,3], [1,2,3]])
    yy = np.array([1,1,2,2])

    xxt = np.array([[0,1,2], [1,1,2]])
    yyt = np.array([1,2])
    return(xx, yy, xxt, yyt)


@unknow_error
def test_diff_train_test(func):
    xx, yy, xxt, yyt = generate_data2()
    m = MLPClassifier(hidden_layer_sizes=[20,20], max_iter=500, alpha =1e-6, random_state=1)
    m.fit(xx, yy)
    res = func(m,xx,yy, xxt, yyt)
    res_ran = func(m,np.random.randn(30,3),np.random.choice(3,30), xxt, yyt)

    tests = {'no estas retornado lo requerido': res == (1.0, 0.5, 0.5),
             'evitar dejar código estatico': res_ran != res}

    code_to_look = ['accuracy_score', 'predict(Xtrain', 'predict(Xtest)', "y_true=Ytrain", "y_true=Ytest", 'abs']
    res2 = ut.check_code(code_to_look, func, "recordar usar los metodos, errores sugeridos y llamar explicitamente los parametros de sklearn")

    return (ut.test_conditions_and_methods(tests) and res2)

@unknow_error
def test_exp_mlp_early_stop(func):
    xx, yy = ut.get_linear_separable_dataset(ext = False)
    xxt, yyt = ut.get_nolinear_separable_dataset(ext = False)
    num_neurons = [4,8,16]
    is_early_stop = [True, False]
    cols =['neuronas en capas ocultas', 'error de entrenamiento',
       'error de prueba', 
       'diferencia entrenamiento y prueba',
       'is_early_stop']

    cols_errs = ['error de entrenamiento',
       'error de prueba']

    res = ut.test_experimento_oneset(func,  shape_val=(len(num_neurons)*len(is_early_stop), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    Xtrain = xx,
                                    Xtest = xxt,
                                    Ytrain = yy , 
                                    Ytest = yyt, 
                                    num_neurons = num_neurons, 
                                    is_early_stop= is_early_stop)
    code_to_look = ['MLPClassifier', 'diff_train_test', 'random_state=1', 
                    'early_stopping=' , ".fit"] 
    res2 = ut.check_code(code_to_look, func)

    return (res and res2)

@unknow_error
def test_exp_mlp_l2(func):
    xx, yy = ut.get_nolinear_separable_dataset(ext = False, random_state = 50)
    xxt, yyt = ut.get_nolinear_separable_dataset(ext = False)
    num_neurons = [4,6]
    l2_values = [1e-8,1e-10]
    cols =['neuronas en capas ocultas', 'error de entrenamiento',
       'error de prueba', 
       'diferencia entrenamiento y prueba',
       'l2']
    cols_errs = ['error de entrenamiento',
       'error de prueba']

    res = ut.test_experimento_oneset(func,  shape_val=(len(num_neurons)*len(l2_values), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    Xtrain = xx,
                                    Xtest = xxt,
                                    Ytrain = yy , 
                                    Ytest = yyt, 
                                    num_neurons = num_neurons, 
                                    l2_values= l2_values)
    code_to_look = ['MLPClassifier', 'diff_train_test', 'random_state=1', 
                    'alpha=' , ".fit"] 
    res2 = ut.check_code(code_to_look, func)

    return (res and res2)


@unknow_error
def test_exp_reg_l2(func):
    xx, yy = ut.get_linear_separable_dataset(ext = False, random_state = 50)
    xxt, yyt = ut.get_nolinear_separable_dataset(ext = False)
    l2_values = [1,10]
    cols =['error de entrenamiento',
       'error de prueba', 
       'diferencia entrenamiento y prueba',
       'l2']
    cols_errs = ['error de entrenamiento',
       'error de prueba']

    res = ut.test_experimento_oneset(func,  shape_val=(len(l2_values), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    Xtrain = xx,
                                    Xtest = xxt,
                                    Ytrain = yy , 
                                    Ytest = yyt, 
                                    l2_values= l2_values)
    code_to_look = ['LogisticRegression', 'diff_train_test', 'random_state=1', 
                    'C=' , ".fit"] 
    res2 = ut.check_code(code_to_look, func)

    return (res and res2)

@unknow_error
def test_train_size_experiments(func):
    xx, yy = ut.get_linear_separable_dataset(ext = False, random_state = 50)
    train_pcts = [0.1,0.8]
    cols =['error de entrenamiento',
       'error de prueba', 
       'diferencia entrenamiento y prueba',
       'tamaño de entrenamiento']
    cols_errs = ['error de entrenamiento',
       'error de prueba']

    m = MLPClassifier(hidden_layer_sizes=[20,20], max_iter=500, alpha =1e-6, random_state=1)

    res, df = ut.test_experimento_oneset(func,  shape_val=(len(train_pcts), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    return_df = True,
                                    X = xx,
                                    Y = yy , 
                                    sk_estimator = m,
                                    train_pcts = train_pcts)
    code_to_look = ['diff_train_test', 'random_state=10'] 
    res2 = ut.check_code(code_to_look, func)

    if not(ut.is_inc_dec(df['tamaño de entrenamiento'].values , increasing = True )):
        print("error, el conjunto de entrenamiento no esta creciendo!", df['tamaño de entrenamiento'].values)
        return (False)

    return (res and res2)



def part_2():
    GRADER = Grader("lab4_part2", num_questions = 5)
    GRADER.add_test("ejercicio1", Tester(test_diff_train_test))
    GRADER.add_test("ejercicio2", Tester(test_exp_mlp_early_stop))
    GRADER.add_test("ejercicio3", Tester(test_exp_mlp_l2))
    GRADER.add_test("ejercicio4", Tester(test_exp_reg_l2))
    GRADER.add_test("ejercicio5", Tester(test_train_size_experiments))
    return(GRADER)
