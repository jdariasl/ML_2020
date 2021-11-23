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

def potenciaPolinomio(X,grado):
    """calcula la potencia del polinomio
    X: los valores que corresponden a las caractersiticas
    grado: esl grado para realizar la potencia al polinomio
    """
    X2 = X.copy()
    
    if grado != 1:
        for i in range(2,grado+1):
            Xadd = X**i
            X2 = np.concatenate((X2, Xadd), axis=1)
    
    return X2


def genarete_data():
    db = np.loadtxt('AirQuality.data',delimiter='\t') 
    xtrain = np.random.rand(100,3)
    ytrain = np.ones(shape = (100,1))
    x = db[0:100,0:4]
    y = db[0:100, 12]
    wr1 = np.zeros((1,x.shape[1]))
    wr1 = wr1.reshape(np.size(wr1), 1)
    wr2 = np.random.rand(1,xtrain.shape[1])
    wr2 = wr2.reshape(np.size(wr2), 1)

    return(xtrain, x, ytrain, y, wr1, wr2)

@unknow_error
def test_ejercicio_1(func):
    nr = 100
    nc = 4
    xtrain1, x, _, _, _ ,_= genarete_data()
    tests = {'Debes retornar el numero de muestras y columnas': func (x) == (nc,nr),
             'Recuerda que la funcion debe recibir la variable parametro':  func (xtrain1) == (3,100) }
    test_res = ut.test_conditions_and_methods(tests)
    return (test_res)

    
@unknow_error
def test_ejercicio_2(func):
    code_to_look = [['.dot', 'cost', 'X_ext.T', 'extension_matriz'], 
                   ['extension_matriz', 'cost', 'np.sum', 'keepdims=True']]
    res2 = ut.check_code(code_to_look, func, "usar solo numpy y funciones previamente definidas")
    
    xx = np.array([[0,0], [1,1], [2,2]])
    yy = np.array([1,2,4]).reshape(3,1)
    ww1, cc1 = func(xx, yy, 0.001, 15)

    xx2 = np.array([[0], [1], [2]])
    yy2 = np.array([0,1,2]).reshape(3,1)
    ww2, cc2 = func(xx2, yy2, 0.001, 15)

    tests_dict =  {'revisar las operaciones':  np.allclose(ww1,np.array([[0.03407114],[0.04861253], [0.04861253]])),
                  'El costo no esta dismuyendo verificar': np.all(np.diff(cc1)<0) or np.all(np.diff(cc2)<0),
                  'Evitar dejar codigo estatico': np.sum(ww1) != np.sum(ww2),
                  'Recuerda las iteraciones': len(cc1)==15}
    
    test  =  ut.test_conditions_and_methods(tests_dict)
    return (res2 and test)

@unknow_error
def test_ejercicio_3(func):
    code_to_look = ['extension_matriz', 'ECM', 'regression']
    code_check = ut.check_code(code_to_look, func, "usar solo numpy y funciones previamente definidas")
    
    ww1 = np.array([[0.5], [0.5], [0.5]])
    xx1 = np.array([[0,0], [1,1], [2,2]])
    yy1 = np.array([0,1,2])
    e1 = func(ww1, xx1, yy1)

    ww2 = np.array([[1.0], [0.0]])
    xx2 = np.array([[0], [1], [2]])
    yy2 = np.array([1,1,1])
    e2 = func(ww2, xx2, yy2)

    tests_dict =  {'revisar las operaciones':  np.allclose(e1, 0.25) and np.allclose(e2, 0.0)}
    test  =  ut.test_conditions_and_methods(tests_dict)
    return (code_check and test)

   
@unknow_error
def test_ejercicio_4(func):
    code_to_look = ['potenciaPolinomio', 'gradiente_descendente(X2']
    code_check = ut.check_code(code_to_look, func, "usar solo numpy y funciones previamente definidas",debug = False)
    
    xx = np.array([[1,1], [-1,-1], [2,2], [-2,-2]])
    yy = np.array([1,0.9,0.5,0.45]).reshape(4,1)

    ww1, cc1 = func(xx, yy, 0.001, 20, 1)
    ww2, cc2 = func(xx, yy, 0.001, 6, 2)
    ww3, _ = func(xx, yy, 0.001, 5, 3)

    tests_dict =  {'revisar las operaciones':  np.allclose(ww1,np.array([[0.01411543], [0.0009539   ], [0.0009539]])),
                  'Evitar dejar codigo estatico': np.sum(ww1) != np.sum(ww2),
                  'Recuerda aplicar el polinomio': ww1.shape == (3,1) and ww2.shape == (5,1) and ww3.shape == (7,1),
                  'Recuerda las iteraciones': len(cc1)==20 and len(cc2)==6}
    test  =  ut.test_conditions_and_methods(tests_dict)
   
    return (code_check and test)


@unknow_error
def test_ejercicio_5(func):
    code_to_look = ['potenciaPolinomio', 'evaluar_modelo(W,X2']
    code_check = ut.check_code(code_to_look, func, "usar solo numpy y funciones previamente definidas")
    ww1 = np.array([[1.0], [0.0]])
    xx1 = np.array([[0], [1], [2]])
    yy1 = np.array([1,1,1])
    e1 = func(ww1, xx1, yy1, 1)
    ww2 = np.array([[1.0], [0.0], [1.0]])
    e2 = func(ww2, xx1, yy1, 2)
    ww3 = np.array([[1.0], [1.0], [1.0], [1.0]])
    e3 = func(ww3, xx1, yy1, 3)
    tests_dict =  {'revisar las operaciones':  np.allclose(e1, 0.0) and np.allclose(e2, 5.666666666666667) and  np.allclose(e3, 68.333333)}
    test  =  ut.test_conditions_and_methods(tests_dict) 
    return (test and code_check)

@unknow_error
def test_exp1(func):
    tasas = [1e-2, 1e-3]
    grados = [1,2,3]
    xtrain, _, ytrain, _, _, _ = genarete_data()

    res = ut.test_experimento(func, xtrain=xtrain, 
                              ytrain=ytrain, 
                              shape_val=(len(tasas)*len(grados), 3),
                              col_val=['grado', 'tasa de aprendizaje', 'ecm'],
                              tasas = tasas, 
                              grados= grados)
    return (res)

@unknow_error
def test_exp2(func):
    grados = [2,5]
    iteraciones = [2,3]
    xtrain, _, ytrain, _, _, _ = genarete_data()

    res = ut.test_experimento(func, xtrain=xtrain, 
                              ytrain=ytrain, 
                              shape_val=(len(iteraciones)*len(grados), 3),
                              col_val = ['iteraciones', 'grado', 'ecm'],
                              iteraciones = iteraciones, 
                              grados= grados)
    return (res)

def part_1():
    print("cargando librerias y variables al ambiente")
    GRADER_LAB_1_P1 = Grader("lab1_part1")
    GRADER_LAB_1_P1.add_test("ejercicio1", Tester(test_ejercicio_1))
    GRADER_LAB_1_P1.add_test("ejercicio2", Tester(test_ejercicio_2))
    GRADER_LAB_1_P1.add_test("ejercicio3", Tester(test_ejercicio_3))
    GRADER_LAB_1_P1.add_test("ejercicio4",  Tester(test_ejercicio_4))
    GRADER_LAB_1_P1.add_test("ejercicio5", Tester(test_ejercicio_5))
    GRADER_LAB_1_P1.add_test("ejercicio6", Tester(test_exp1))
    GRADER_LAB_1_P1.add_test("ejercicio7", Tester(test_exp2))
    db = np.loadtxt('AirQuality.data',delimiter='\t') 
    x = db[:,0:12]
    y = db[:,12]
    return (GRADER_LAB_1_P1, db, x, y)

#
# parte 2
#

def genarete_data2():
    mat = scipy.io.loadmat('DatosClases.mat')
    x = mat['X'] # Matriz X de muestras con las características
    y = mat['Y'] # Variable de salida
    xtrain = np.random.rand(100,2)
    ytrain = np.ones(shape = (100,1))
    wr1 = np.zeros((1,x.shape[1]))
    wr1 = wr1.reshape(np.size(wr1), 1)
    wr2 = np.random.rand(1,xtrain.shape[1])
    wr2 = wr2.reshape(np.size(wr2), 1)

    return(xtrain, x, ytrain, y, wr1, wr2)


@unknow_error
def test_ejercicio_1_p2(func):
    xtrain, x, ytrain, y, _ ,_= genarete_data2()
    tests = {'Debes retornar el numero de muestras y columnas': func (x, y) == (2, 500, 2),
             'Recuerda que la funcion debe recibir la variable  como parametro':  func (xtrain, ytrain) == (1,100,2) }
    test_res = ut.test_conditions_and_methods(tests)
    return (test_res)

@unknow_error
def test_ejercicio_2_p2(func):
    xtrain, x, ytrain, y, _ ,_= genarete_data2()
    code_to_look = [['scatter', 'X[:,0]', "X[:,1]", "c=Y", "cmap="], ]
    res2 = ut.check_code(code_to_look, func, 'revisar la funcion que se uso de plt')
    tests = {'revisa tu implementacion': ut.work_well(func, X=xtrain, Y=ytrain)}

    test_res = ut.test_conditions_and_methods(tests)

    return (res2 and test_res)

@unknow_error
def test_sigmoide(func):
    t1 = ut.are_np_equal(func(np.zeros(2)), np.array([0.5, 0.5]))
    xx = np.random.rand(5)
    t2 = ut.are_np_equal(func(xx),  1/(1+np.exp(-(xx))))
    tests = {'revisa tu implementacion': t1,
             'revisa tu implementacion, recuerda evitar código estatico': t2}
    test_res = ut.test_conditions_and_methods(tests)
    return (test_res)


@unknow_error
def test_gradiente_descendente_logistic_poly(func):
    code_to_look = ['extension_matriz', 'logistic_regression', 'cost_logistic']
    code_check = ut.check_code(code_to_look, func, "usar solo numpy y funciones previamente definidas")
    xx1 = np.array([[1,1],[-1,-1], [2,2], [-2,-2]])
    yy1 = np.array([[1], [0],[1],[0]])
    yy2 = np.array([[1], [1],[0],[0]])
    w1, cl1 = func(xx1,yy1,grado = 1, eta = 1.0, iteraciones = 10)
    w2, cl2 = func(xx1,yy2,grado = 2, eta = 1.0, iteraciones = 30)
    
   
    tests_dict =  {'revisar las operaciones':  np.allclose(w1, np.array([[-6.93889390e-17], [1.36689086], [1.36689086]])),
                   'cuidado el costo parece no disminuir': np.all(np.diff(cl1)<0) and np.all(np.diff(cl2[1:])<0),
                   'recuerda realizar extension de matrices y aplicar el polinomio': w1.shape == (3,1) and w2.shape == (5,1),
                   'recuerda las iteraciones': len(cl1) == 10 and len(cl2) == 30
                   }
    test  =  ut.test_conditions_and_methods(tests_dict) 
    return (test and code_check)

@unknow_error
def test_exp1_part2(func):
    tasas = [1,0.1]
    grados = [1,2,3]
    xl,yl = ut.get_nolinear_separable_dataset()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(xl, yl, test_size=0.2, random_state=10)
    res = ut.test_experimento_train_test(func, xtrain=X_train, xtest=X_test,
                              ytrain=y_train, ytest = y_test,
                              shape_val=(len(tasas)*len(grados), 4),
                              col_val=['grado','tasa de aprendizaje', 'error_entreamiento'	, 'error_prueba'],
                              col_error = ['error_entreamiento'	, 'error_prueba'],
                              tasas = tasas, 
                              grados= grados)
    code_to_look = ['evaluar_modelo']
    res2 = ut.check_code(code_to_look, func, "recuerda usar las funciones anteriores!")
                                 
    return (res and res2)

def part_2():
    print("cargando librerias y variables al ambiente")
    GRADER = Grader("lab1_part2")
    GRADER.add_test("ejercicio1", Tester(test_ejercicio_1_p2))
    GRADER.add_test("ejercicio2", Tester(test_ejercicio_2_p2))
    GRADER.add_test("ejercicio3", Tester(test_sigmoide))
    GRADER.add_test("ejercicio4", Tester(test_gradiente_descendente_logistic_poly))
    GRADER.add_test("ejercicio5",  Tester(test_exp1_part2))
    #GRADER.add_test("ejercicio6", Tester(test_exp1))
    #GRADER.add_test("ejercicio7", Tester(test_exp2))
    mat = scipy.io.loadmat('DatosClases.mat')
    x = mat['X'] # Matriz X de muestras con las características
    y = mat['Y'] # Variable de salida
    return (GRADER, x, y)

def normalizar(Xtrain, Xtest):
    """ función que se usa para normalizar los datos con
    un metodo especifico
    Xtrain: matriz de datos entrenamiento a normalizar
    Xtest: matriz de datos evaluación a normalizar
    retorna: matrices normalizadas
    """
    media = np.mean(Xtrain, axis = 0)
    desvia = np.std(Xtrain, axis = 0)
    Xtrain_n = stats.stats.zscore(Xtrain)
    Xtest_n = (Xtest - media )/desvia
    # si hay una desviacion por cero, reemplazamos los nan
    Xtrain_n = np.nan_to_num(Xtrain_n)
    Xtest_n = np.nan_to_num(Xtest_n)
    return (Xtrain_n, Xtest_n)
