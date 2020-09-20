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
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def generate_data(is_class = False):
    yy = np.random.choice(2, 30) if is_class else 2*np.random.rand(60).reshape((30,2))
    xx = np.vstack([np.random.rand(15, 3), 2*np.random.rand(15, 3)])
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
    code_to_look = ['MLPRegressor', 'hidden_layer_sizes=', 'activation=', "'tanh'",  
                    'max_iter=' , ".fit", ".predict(Xtest)", "X=Xtrain,", 
                    "hidden_layers*[neurons]", "mean_absolute_error", 'multioutput=',
                    "np.mean(ErrorY1)", "np.mean(ErrorY2)"] 
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
    xx, yy = generate_data(True)
    capas = [1,2]
    neu = [5,10]
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
    code_to_look = ['MLPClassifier', 'hidden_layer_sizes=', 'activation=', "'tanh'",  
                    'max_iter=' , ".fit", ".predict(Xtest)", "X=Xtrain,",  
                     'accuracy_score', "hidden_layers*[neurons]"] 
    res2 = ut.check_code(code_to_look, func)
    return (res and res2)

def part_1 ():
    GRADER = Grader("lab4_part1", num_questions = 4)
    GRADER.add_test("ejercicio1", Tester(test_output_activation))
    GRADER.add_test("ejercicio2", Tester(test_experimetar_mlp))
    GRADER.add_test("ejercicio3", Tester(test_output_activation_MPC))
    GRADER.add_test("ejercicio4", Tester(test_experimetar_mlpc))
    return(GRADER)
