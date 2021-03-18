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
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)


@unknow_error
def test_get_muestras_by_cv(func):

    Y = np.random.choice(3,100)
    cv1 = func(np.ones((100,2)),Y, 1)
    cv2= func(np.ones((100,2)),Y, 2)  
    met = cv2['numero de muestras entrenamiento'].mean() <= cv1['numero de muestras entrenamiento'].mean()

    tests = {'recuerda dividir en 4 folds': (cv1.shape[0] == len(np.unique(Y))*4 ) and (cv2.shape[0] == len(np.unique(Y))*4) ,
             'recuerda que metodo corresponde a la metodologia de validacion': met
             }
    test_res = ut.test_conditions_and_methods(tests)
 
    res = ut.test_experimento_oneset(func,  shape_val=(len(np.unique(Y))*4, 3), 
                                    col_error = ['etiqueta de clase'],
                                    col_val=['etiqueta de clase', 'folds', 'numero de muestras entrenamiento'],
                                    X = np.ones((100,2)), Y=Y,
                                    method = 1)
    return (res and test_res )

@unknow_error
def test_GMMClassifierTrain(func):
    y1 = np.random.choice(3,20)
    g1 = func(np.random.rand(20,2), y1, 2, 'full')
    g2 = func(np.random.rand(20,2), np.random.choice(3,20), 2, 'diag')
    g3 = func(np.random.rand(20,2), np.random.choice(3,20), 2, 'tied')
    g4 = func(np.random.rand(10,2), np.random.choice(2,10), 2, 'spherical')
    t1 =  len(np.array([np.mean(m.means_) for m in g1.values()])) == len(np.unique([np.mean(m.means_) for m in g1.values()]))

    tests = {'debes entrenar un GMM por cada clase (valor unico de Y)': t1,
             'la clave del dict debe ser la etiqueta de Y':  (list(g1.keys()) == np.unique(y1)).all(),
             'no debes dejear codigo estatico.':  g1 != g2 }

    return (ut.test_conditions_and_methods(tests))

@unknow_error
def test_GMMClassfierVal(func):
    yy = np.random.choice(2, 20)
    xx = np.random.rand(20, 3)
    xx2 = np.random.rand(10, 3)
    yy2 = np.zeros(10)
    gmms = {0: GaussianMixture().fit(xx[yy==0]),
            1: GaussianMixture().fit(xx[yy==1])}
    gmms2 = {0: GaussianMixture().fit(xx2)}

    yy_res, probs = func(gmms, xx)
    _, probs2 = func(gmms2, xx2)

    tests = {'debes retornar las probabilidades por cada clase': len(np.unique(probs.sum(axis =1))) == len(probs.sum(axis =1)),
             'la salida debe la etiqueta de las clases predichas':  (np.unique(yy_res) == np.unique(yy)).all(),
             'el shape de la matriz de probs es incorrecto': yy_res.shape[0] == xx.shape[0],
             'evita dejar codigo estatico':  probs.shape == (xx.shape[0], len(np.unique(yy))) and probs2.shape == (xx2.shape[0], len(np.unique(yy2)))
             }
    return (ut.test_conditions_and_methods(tests))

@unknow_error
def test_experimentar(func):
    yy = np.random.choice(2, 30)
    xx = np.vstack([np.random.rand(15, 3), 2*np.random.rand(15, 3)])
    mts = ['full', 'tied', 'diag', 'spherical']
    ms = [1,2,3]
    cols = ['matriz de covarianza','numero de componentes',
            'eficiencia de entrenamiento',
            'desviacion estandar entrenamiento',
            'eficiencia de prueba',
            'desviacion estandar prueba']

    errs = ['eficiencia de entrenamiento',
            'eficiencia de prueba']

    res = ut.test_experimento_oneset(func,  shape_val=(len(mts)*len(ms), 6), 
                                    col_error = errs,
                                    col_val=cols,
                                    X = xx, Y=yy,
                                    covariance_types = mts,
                                    num_components = ms)
    return (res)

@unknow_error
def test_experimentar_kmeans(func):
    code_to_look = [['KMeans', ".fit", ".predict", 'init="k-means++"'], 
                    ['KMeans', ".fit", ".predict", "init='k-means++'"]]
    res2 = ut.check_code(code_to_look, func)
    yy = np.random.choice(2, 30)
    xx = np.vstack([np.random.rand(15, 3), 2*np.random.rand(15, 3)])
    nc = [1,2,3]
    cols = ['numero de clusters',
            'eficiencia de entrenamiento',
            'desviacion estandar entrenamiento',
            'eficiencia de prueba',
            'desviacion estandar prueba']

    errs = ['eficiencia de entrenamiento',
            'eficiencia de prueba']

    res = ut.test_experimento_oneset(func, shape_val=(len(nc), 5), 
                                    col_error = errs,
                                    col_val=cols,
                                    X = xx, Y=yy,
                                    numero_clusters = nc)
    
    return (res and res2)

def part_1 ():
    #cargamos la bd iris desde el dataset de sklearn
    GRADER = Grader("lab3_part1")
    GRADER.add_test("ejercicio1", Tester(test_get_muestras_by_cv))
    GRADER.add_test("ejercicio2", Tester(test_GMMClassifierTrain))
    GRADER.add_test("ejercicio3", Tester(test_GMMClassfierVal))
    GRADER.add_test("ejercicio4", Tester(test_experimentar))
    GRADER.add_test("ejercicio5", Tester(test_experimentar_kmeans))
    return(GRADER)


cols_errs = ['eficiencia de entrenamiento','eficiencia de prueba']

def generate_data():
    yy = np.random.choice(2, 30)
    xx = np.vstack([np.random.rand(15, 3), 2*np.random.rand(15, 3)])
    return (xx, yy)

@unknow_error
def test_experimentar_dt(func):
    xx, yy = generate_data()
    depths = [3,5,10]
    cols = ['profunidad del arbol', 'eficiencia de entrenamiento',
            'desviacion estandar entrenamiento', 'eficiencia de prueba',
            'desviacion estandar prueba']
    res = ut.test_experimento_oneset(func,  shape_val=(len(depths), 5), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    X = xx, Y=yy,
                                    depths = depths,
                                    normalize = True)
    code_to_look = ['DecisionTreeClassifier', 'max_depth=', ".fit", ".predict"]
    res2 = ut.check_code(code_to_look, func)

    return (res and res2)

@unknow_error
def test_experimentar_rf(func):
    xx, yy = generate_data()
    trees = [3,5,10]
    num_vars = [1,2,3]
    cols = ['número de arboles', 'variables para la selección del mejor umbral',
       'eficiencia de entrenamiento', 'desviacion estandar entrenamiento',
       'eficiencia de prueba', 'desviacion estandar prueba']
   
    res = ut.test_experimento_oneset(func,  shape_val=(len(trees)*len(num_vars), 6), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    X = xx, Y=yy,
                                    num_trees = trees,
                                    numero_de_variables = num_vars)
    code_to_look = ['RandomForestClassifier', 'n_estimators=', 'max_features=',  ".fit", ".predict"]
    res2 = ut.check_code(code_to_look, func)
    return (res and res2)

@unknow_error
def test_experimentar_gbt(func):
    xx, yy = generate_data()
    trees = [3,5,10]
    cols = ['número de arboles', 'eficiencia de entrenamiento',
       'desviacion estandar entrenamiento', 'eficiencia de prueba',
       'desviacion estandar prueba']
    res = ut.test_experimento_oneset(func,  shape_val=(len(trees), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    X = xx, Y=yy,
                                    num_trees = trees)
    code_to_look = ['GradientBoostingClassifier', 'n_estimators=',  ".fit", ".predict"]
    res2 = ut.check_code(code_to_look, func)
    return (res and res2)

@unknow_error
def test_time_rf_training(func):
    xx, yy = generate_data()
    trees = [3,5,10]
    num_vars = [1,2,3]
    cols = ['número de arboles', 'variables para la selección del mejor umbral',
       'tiempo de entrenamiento']
    res = ut.test_experimento_oneset(func,  shape_val=(len(trees)*len(num_vars), len(cols)), 
                                    col_error = ['tiempo de entrenamiento'],
                                    col_val=cols,
                                    X = xx, Y=yy,
                                    num_trees = trees,
                                    numero_de_variables = num_vars)
    code_to_look = ['RandomForestClassifier', 'n_estimators=', "max_features=", "time.clock()",  ".fit"]
    res2 = ut.check_code(code_to_look, func)
    return (res and res2)


def part_2 ():
    #cargamos la bd iris desde el dataset de sklearn
    GRADER = Grader("lab3_part2")
    GRADER.add_test("ejercicio1", Tester(test_experimentar_dt))
    GRADER.add_test("ejercicio2", Tester(test_experimentar_rf))
    GRADER.add_test("ejercicio3", Tester(test_experimentar_gbt))
    GRADER.add_test("ejercicio4", Tester(test_time_rf_training))
    return(GRADER)