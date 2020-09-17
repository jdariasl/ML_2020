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
    tests = {'Debes retornar el numero de muestras y columnas': func (x) == (nr,nc),
             'Recuerda que la funcion debe recibir la variable parametro':  func (xtrain1) == (100,3) }
    test_res = ut.test_conditions_and_methods(tests)
    return (test_res)

    
@unknow_error
def test_ejercicio_2(func):
    xtrain1, x, ytrain, y, wr1, _ = genarete_data()
    def ww (w):
        some = (np.sum(x*(np.dot(x,w) - y.reshape((x.shape[0],1))), axis = 0, keepdims = True)/((x.shape[0]))) .T
        return (w-0.001*some)
    wr1= ww(wr1)
    wr1= ww(wr1)
    w1 = func(x, y, 0.001, 2)
    w2 = func(xtrain1, ytrain, 0.001, 2)
    tests = {'revisa tu implementacion. \n Sigue la instrucciones ten cuidado con las dimensiones de la matrices. \n evita dejar codigo estatico ': 
              ut.are_np_equal(w1,wr1),
             'Recuerda que la funcion debe recibir parametros, evita dejar codigo estatico':  w2.shape == (xtrain1.shape[1],1) }
    test_res = ut.test_conditions_and_methods(tests)
    return (test_res)

@unknow_error
def test_ejercicio_3(func):
    xtrain1, x, ytrain, y, wr1, wr2 = genarete_data()
    est = np.dot(x,wr1)
    te = np.sum((est.reshape(y.shape[0],1) - y.reshape(y.shape[0],1))**2)/(y.shape[0])

    est2 = np.dot(xtrain1,wr2)
    te2 = np.sum((est2.reshape(ytrain.shape[0],1) - ytrain.reshape(ytrain.shape[0],1))**2)/(ytrain.shape[0])

    error = func(wr1, X_to_test = x,  Y_True = y)
    error2 = func(wr2, X_to_test = xtrain1,  Y_True = ytrain)

    tests = {'revisa tu implementacion. \n Sigue las instrucciones. \n evita dejar codigo estatico ': te == error,
             'Recuerda que la funcion debe recibir parametros, evita dejar codigo estatico':  te2 == error2 }
    test_res = ut.test_conditions_and_methods(tests)
    return (test_res)

   
@unknow_error
def test_ejercicio_4(func):
    xtrain1, x, ytrain, y, _, _ = genarete_data()
    x_g2 = potenciaPolinomio(x, 3)
    xtrain_g2 = potenciaPolinomio(xtrain1,3)
    wr1 = np.zeros((1,x_g2.shape[1]))
    wr1 = wr1.reshape(np.size(wr1), 1)

    def ww (w):
        some = (np.sum(x_g2*(np.dot(x_g2,w) - y.reshape((x.shape[0],1))), axis = 0, keepdims = True)/((x_g2.shape[0]))) .T
        return (w-0.001*some)
    wr1= ww(wr1)
    wr1= ww(wr1)
    w1 = func(x, y, 0.001, 2, 3)
    w2 = func(xtrain1, ytrain, 0.001, 2,3)
    tests = {'revisa tu implementacion. \n Sigue la instrucciones ten cuidado con las dimensiones de la matrices. \n evita dejar codigo estatico ': 
              ut.are_np_equal(w1,wr1),
             'Recuerda que la funcion debe recibir parametros, evita dejar codigo estatico':  w2.shape == (xtrain_g2.shape[1],1) }
    test_res = ut.test_conditions_and_methods(tests)
    return (test_res)


@unknow_error
def test_ejercicio_5(func):
    xtrain1, x, ytrain, y, _, _ = genarete_data()
    x_g = potenciaPolinomio(x, 3)
    xtrain_g = potenciaPolinomio(xtrain1,3)
    wr1 = np.zeros((1,x_g.shape[1]))
    wr1 = wr1.reshape(np.size(wr1), 1)

    wr2 = np.zeros((1,xtrain_g.shape[1]))
    wr2 = wr2.reshape(np.size(wr2), 1)

    est = np.dot(x_g,wr1)
    te = np.sum((est.reshape(y.shape[0],1) - y.reshape(y.shape[0],1))**2)/(y.shape[0])

    est2 = np.dot(xtrain_g,wr2)
    te2 = np.sum((est2.reshape(ytrain.shape[0],1) - ytrain.reshape(ytrain.shape[0],1))**2)/(ytrain.shape[0])

    error = func(wr1, X_to_test = x,  Y_True = y, grado = 3)
    error2 = func(wr2, X_to_test = xtrain1,  Y_True = ytrain, grado = 3)
    tests = {'revisa tu implementacion. \n Sigue las instrucciones. \n evita dejar codigo estatico ': te == error,
             'Recuerda que la funcion debe recibir parametros, evita dejar codigo estatico':  te2 == error2 }
    test_res = ut.test_conditions_and_methods(tests)
    return (test_res)

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

# parte 2
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
    tests = {'Debes retornar el numero de muestras y columnas': func (x, y) == (500, 2, 2),
             'Recuerda que la funcion debe recibir la variable  como parametro':  func (xtrain, ytrain) == (100,2,1) }
    test_res = ut.test_conditions_and_methods(tests)
    return (test_res)

@unknow_error
def test_ejercicio_2_p2(func):
    xtrain, x, ytrain, y, _ ,_= genarete_data2()

    for xx, yy, cl in zip ([xtrain, x], [ytrain, y], [1,2]):
        f = func(xx, yy)
        t1 = (len(f.axes)) == 1 and len(f.axes[0].collections)>= cl
        t2 = ut.are_np_equal(ut.get_data_from_scatter(f.axes[0]), ut.get_org_data(xx,yy))

        if t2 : 
            tests = {'Recuerda que debes graficas las dos clases': t1,
                    'Recuerda que la debe graficar los valores de x y y. y debe ser los colores':  t2}
            test_res = ut.test_conditions_and_methods(tests)
            return (test_res)
        else:
            code_to_look = ['scatter', 'X[:,1]', "X[:,2]", "c=Y"]
            res2 = ut.check_code(code_to_look, func)
            return (res2)

@unknow_error
def test_sigmoide(func):
    t1 = ut.are_np_equal(func(np.zeros(2)), np.array([0.5, 0.5]))
    xx = np.random.rand(5)
    t2 = ut.are_np_equal(func(xx),  1/(1+np.exp(-(xx))))
    tests = {'revisa tu implementacion': t1,
             'revisa tu implementacion, recuerda evitar código estatico': t2}
    test_res = ut.test_conditions_and_methods(tests)
    return (test_res)

def logistic_regression(X, W):
    """calcula la regresión logistica
    X: los valores que corresponden a las caractersiticas
    W: son los pesos usadados para realizar la regresión
    retorna: valor estimado por la regresion
    """
    #Con np.dot se realiza el producto matricial. Aquí X (extendida) tiene dim [Nxd] y W es dim [dx1]
    Yest = np.dot(X,W)
    Y_lest = 1/(1 + np.exp(-Yest))
    #Se asignan los valores a 1 o 0 según el modelo de regresión logística definido
    pos = 0
    for tag in Y_lest:
        
        if tag > 0.5:
            Y_lest[pos] = 1
        elif tag < 0.5:
            Y_lest[pos] = 0
        
        pos += 1
    
    return Y_lest    #Y estimado: Esta variable contiene ya tiene la salida de sigm(f(X,W))

@unknow_error
def test_gradiente_descendente_logistic_poly(func):
    xtrain, x, ytrain, y, wr1, wr2 = genarete_data2()
    x_g = potenciaPolinomio(x,3)
    wr2 = np.zeros((1,x_g.shape[1]))
    wr2 = wr2.reshape(np.size(wr2), 1)
   
    wr1 = wr1-0.01*(np.dot(x.T, logistic_regression(x,wr1) - y))/x.shape[0]
    wr1 = wr1-0.01*(np.dot(x.T, logistic_regression(x,wr1) - y))/x.shape[0]
    error = func(x,y,1,0.01, 2)
    t1 = ut.are_np_equal(wr1, error)

    wr2 = wr2-0.01*(np.dot(x_g.T, logistic_regression(x_g,wr2) - y))/x_g.shape[0]
    wr2 = wr2-0.01*(np.dot(x_g.T, logistic_regression(x_g,wr2) - y))/x_g.shape[0]

    tests = {'revisa tu implementacion, el test 1 fallo ': t1,
             'revisa tu implementacion, el test 2 fallo':  ut.are_np_equal(wr2, func(x,y,3,0.01, 2)) }
    test_res = ut.test_conditions_and_methods(tests)

    return (test_res)

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
                              tasas = tasas, 
                              grados= grados)
                                 
    return (res)

@unknow_error
def test_numero_de_errores(func):
    xtrain, x, ytrain, y, wr1, wr2 = genarete_data2()
    x_g = potenciaPolinomio(x,4)
    wr2 = np.ones((x_g.shape[1], 1))
    wr2 = wr2.reshape(np.size(wr2), 1)

    tests = {'revisa tu implementacion, el test 1 fallo ':func (wr1, x, y, 1) == 500,
             'revisa tu implementacion, el test 2 fallo':  func (wr2, x, y, 4) == 244 }
    test_res = ut.test_conditions_and_methods(tests)
                                 
    return (test_res)

def part_2():
    print("cargando librerias y variables al ambiente")
    GRADER = Grader("lab1_part2")
    GRADER.add_test("ejercicio1", Tester(test_ejercicio_1_p2))
    GRADER.add_test("ejercicio2", Tester(test_ejercicio_2_p2))
    GRADER.add_test("ejercicio3", Tester(test_sigmoide))
    GRADER.add_test("ejercicio4", Tester(test_gradiente_descendente_logistic_poly))
    GRADER.add_test("ejercicio5",  Tester(test_exp1_part2))
    GRADER.add_test("ejercicio6", Tester(test_numero_de_errores))
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
