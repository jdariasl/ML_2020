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
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import warnings
import sklearn   
warnings.filterwarnings('ignore')


def generate_data(is_class = False, cols = 2):
    yy = np.random.choice(2, 30) if is_class else 2*np.random.rand(60).reshape((int(60/cols),cols))
    xx = np.vstack([np.random.rand(15, 3), 2*np.random.rand(15, 3)])
    return (xx, yy)


@unknow_error

def test_entrenamiento_sin_seleccion_caracteristicas(function):
    code_to_look = ['KFold(n_splits=', 'accuracy_score', '.predict', '.fit']
    res2 = ut.check_code(code_to_look, function, msg = "**** recordar usar las funciones sugeridas ***", debug = False)
    xx, yy = generate_data(is_class=True)
    a = function(3,xx,yy)
    tests = {'*** No se retorno el modelo entrenado ***': isinstance(a[0], sklearn.svm._classes.SVC) ,
             '*** No se retorno el error ***': a[1]>0.25,
             '*** No se retorno el desviacion estandar del error ***':  isinstance(a[2], float) and a[1]!=a[2],
             "*** No se retorno el tiempo de entrenamiento***": isinstance(a[3], float) and a[3]<=0.1
             }

    test_res = ut.test_conditions_and_methods(tests)
    return(test_res and res2)




@unknow_error
def test_recursive_feature_elimination_wrapper(function):
  """
  Esta función determina si el estudiante logró utilizar la función planteada
  """
  code_to_look = ['estimator=estimator', 'n_features_to_select=feature_numbers', 
                 'step=1', '.fit(X=X', '.support_', '.ranking_', '.estimator_']
  res2 = ut.check_code(code_to_look, function, msg = "**** recordar usar las funciones sugeridas ***", debug = False)

  xx =  np.array([[-1,0,0.0], [-1,-1,0.0], [1,1,0.1], [1,0.5,0.1], [0.1,0.1,0]])
  yy = np.array([1,1,2,2,2])
  a = function(SVC(kernel="linear", C=1),1,xx, yy)

  tests = {'*** No se retorno el modelo entrenado ***': isinstance(a[0], sklearn.feature_selection._rfe.RFE) ,
          '*** debes retornar la mascara del modelo RFE ***': a[1].sum() == 1,
          '*** retornar el ranking de los features ***':  (a[2] == np.array([1,2,3])).all(),
          "*** No se retorno el modelo interno***": isinstance(a[3], sklearn.svm._classes.SVC),
          "*** No se retorno el tiempo de entrenamiento ***": isinstance(a[4], float) and a[4]<=0.1
            }

  test_res = ut.test_conditions_and_methods(tests)
  return(test_res and res2)


@unknow_error
def test_experimentar(function):
  """
  Esta función determina si el estudiante logró plantear un algoritmo que 
  identifique las propiedades de ejecución de cada "Sabor" del modelo.

  Presume que se retorna un DataFrame cuyas columnas se llamen según lo propuesto
  en el ejercicio.  Se compara con el DataFrame solución construido en esta 
  función, y se retorna True si estos coinciden
  """
  xx, yy = generate_data(is_class=True)
  n_feats = [1,2]
  n_sets = [3,5]
  cols = ['CON_SEL', 'NUM_VAR', 'NUM_SPLITS', 'ERROR_VALIDACION',
       'IC_STD_VALIDACION', 'T_EJECUCION']
  cols_errs = ['ERROR_VALIDACION', 'IC_STD_VALIDACION']

  res = ut.test_experimento_oneset(function,  shape_val=(len(n_feats)*len(n_sets)+len(n_sets), len(cols)), 
                                   col_error = cols_errs,
                                   col_val=cols,
                                   X = xx,
                                   Y = yy,
                                   n_feats = n_feats,
                                   n_sets= n_sets)
  code_to_look = ['entrenamiento_sin_seleccion_caracteristicas', 'recursive_feature_elimination_wrapper', 
                  'feature_numbers=f', '.predict(X_test)', 'mean', '.std', 'n_splits='] 
  res2 = ut.check_code(code_to_look, function, msg = "**** recordar usar las funciones sugeridas ***", debug = False)

  return res2 and res


def part_1 ():
    GRADER = Grader("lab6_part1", num_questions = 4)
    db = np.loadtxt('DB_Fetal_Cardiotocograms.txt',delimiter='\t')  # Assuming tab
    X = db[:,0:22]
    #Solo para dar formato a algunas variables
    for i in range(1,7):
        X[:,i] = X[:,i]*1000
    X = X
    Y = db[:,22]
    Y_l = []
    for i in Y:
        Y_l.append(int(i))
    Y = np.asarray(Y_l)
    GRADER.add_test("ejercicio1", Tester(test_entrenamiento_sin_seleccion_caracteristicas))
    GRADER.add_test("ejercicio2", Tester(test_recursive_feature_elimination_wrapper))
    GRADER.add_test("ejercicio3", Tester(test_experimentar))
    return(GRADER, X, Y)


@unknow_error
def test_entrenamiento_pca_seleccion_caracteristicas(function):
  return True

@unknow_error
def test_pca_benchmark(function):
  return True


def part_2 ():
    GRADER = Grader("lab6_part2", num_questions = 4)
    db = np.loadtxt('DB_Fetal_Cardiotocograms.txt',delimiter='\t')  # Assuming tab
    X = db[:,0:22]
    #Solo para dar formato a algunas variables
    for i in range(1,7):
        X[:,i] = X[:,i]*1000
    X = X
    Y = db[:,22]
    Y_l = []
    for i in Y:
        Y_l.append(int(i))
    Y = np.asarray(Y_l)
    GRADER.add_test("ejercicio5", Tester(test_entrenamiento_pca_seleccion_caracteristicas))
    GRADER.add_test("ejercicio6", Tester(test_pca_benchmark))

    return(GRADER, X, Y)