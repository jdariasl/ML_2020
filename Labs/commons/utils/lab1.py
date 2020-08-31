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
        some = np.dot(x.T, (np.sum((np.dot(x,w)- y)**2, axis = 1, keepdims = True)/(2*( x.shape[0]))))
        return (w-0.001*some/(x.shape[0]))
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
    te = np.sum((est.reshape(y.shape[0],1) - y.reshape(y.shape[0],1))**2)/(2*y.shape[0])

    est2 = np.dot(xtrain1,wr2)
    te2 = np.sum((est2.reshape(ytrain.shape[0],1) - ytrain.reshape(ytrain.shape[0],1))**2)/(2*ytrain.shape[0])

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
        some = np.dot(x_g2.T, (np.sum((np.dot(x_g2,w)- y)**2, axis = 1, keepdims = True)/(2*( x_g2.shape[0]))))
        return (w-0.001*some/(x_g2.shape[0]))
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
    te = np.sum((est.reshape(y.shape[0],1) - y.reshape(y.shape[0],1))**2)/(2*y.shape[0])

    est2 = np.dot(xtrain_g,wr2)
    te2 = np.sum((est2.reshape(ytrain.shape[0],1) - ytrain.reshape(ytrain.shape[0],1))**2)/(2*ytrain.shape[0])

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
