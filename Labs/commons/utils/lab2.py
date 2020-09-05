"""
Este archivo es generado automaticamente.

###### NO MODIFICAR #########

# cualquier alteración del archivo
# puede generar una mala calficacion o configuracion
# que puede repercutir negativamente en la 
# calificacion del laboratorio!!!!!

"""
from imports import *
from scipy.stats import mode
from sklearn.model_selection import train_test_split, StratifiedKFold
from numpy import random
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist

@unknow_error
def test_muestras_por_clases(func):
    ones = func( np.ones(10))
    five = func( np.array([10,20,30,40,50]))

    tests = {'recuerda retornar un diccionario': isinstance(ones, dict) and isinstance(five, dict),
             'el numero de clases es equivalente a a los valores unicos en Y': len(ones) == 1 and len(five) == 5,
             'recuerda que las claves deben ser las etiquetas': list(ones.keys()) == [1] and list(five.keys()) == [10,20,30,40,50]
             }
    test_res = ut.test_conditions_and_methods(tests)
    return (test_res)

@unknow_error
def test_KNN_Clasificacion(func):
   
    xtrains = np.array([[0,1], [1,1], [-1,1], [-1,0], [-0.9,0.15]])
    ytrains =  np.array([0,0,1,1,1])
    xtests = np.array([[0.01,0.9], [0.9,0.9], [-0.99,0.99], [-0.8,-0.1]])

    ytests_should2 = np.array([0., 0., 1., 1.])
    ytests_should3 = np.array([0., 0., 1., 1.])
    ytests_should5 = np.array([1., 1., 1., 1.])
    dists =  np.array([[0.10049876, 0.99503769, 1.01493842, 1.35281189, 1.17923704],
                      [0.90553851, 0.14142136, 1.90262976, 2.1023796 , 1.95      ],
                      [0.9900505 , 1.99002513, 0.01414214, 0.9900505 , 0.84480767],
                      [1.36014705, 2.10950231, 1.11803399, 0.2236068 , 0.26925824]])
    
    lr1, dr1 =  func(np.random.rand(10,2),  np.random.choice(2,10), np.random.rand(3,2), 5)
    lr2, dr2 =  func(np.random.rand(10,2),  np.random.choice(2,10), np.random.rand(3,2), 5)
    r2 =  func(xtrains, ytrains, xtests, 2)
    r3 =  func(xtrains, ytrains, xtests, 3)
    r5 =  func(xtrains, ytrains, xtests, 5)
    t2 =  ut.are_np_equal(r2[1], dists) and ut.are_np_equal(r2[0], ytests_should2) 
    t3 =  ut.are_np_equal(r3[1], dists) and ut.are_np_equal(r3[0], ytests_should3) 
    t5 =  ut.are_np_equal(r5[1], dists) and ut.are_np_equal(r5[0], ytests_should5)
    tr1 = ut.are_np_equal(dr1, dr2, True) and (dr1.shape == (3,10))
    #
    tests = {'test 1 fallo': t2,
             'test 2 fallo': t3,
             'test 3 fallo': t5,
             'test 4 fallo': tr1,
             
             }
    test_res = ut.test_conditions_and_methods(tests)

    return (test_res)

@unknow_error
def test_train_test_split_fix(func):
    x1,x2,y1,y2 = func(np.random.rand(10,2),np.random.choice(2,10))
    x11,x21,y11,y21 = func(np.random.rand(50,3),np.random.choice(2,50))

    tests = {'test 1 fallo': x1.shape == (7,2) and x2.shape == (3,2) and y1.shape[0] == 7 and y2.shape[0] == 3,
             'test 2 fallo': x11.shape == (35,3) and x21.shape == (15,3) and y11.shape[0] == 35 and y21.shape[0] == 15 
            }
    return (ut.test_conditions_and_methods(tests))

@unknow_error
def test_exp(func):
    xx = np.array([[0,1], [1,1], [-1,1], [-1,0], [-0.9,0.15], [0.01,0.9], [0.9,0.9], [-0.99,0.99], [-0.8,-0.1]])
    yy =  np.array([0,0,1,1,1, 1,1,1,1])
    ks = [1,2,3,4]  
    res = ut.test_experimento_oneset(func,  shape_val=(len(ks), 2), 
                                    col_error = ['error de prueba'],
                                    col_val=['k-vecinos', 'error de prueba'],
                                    X = xx, Y=yy,
                                    ks = ks)
    return (res)

@unknow_error
def test_parzenClass(func):
   
    xtrains = np.array([[0.1,0.1], [0.2,0.19], [0.75,0.85],  [1,1], [0.1,0.8], [0.9,0.15]])
    ytrains =  np.array([0,0,2,2,1,1])
    xtests = np.array([[0.25,0.35], [0.9,0.9], [0.1,0.9]])

    ytests_should1 = np.array([0., 2., 1.])
    ytests_should2 = np.array([0., 2., 0.])
    fdp1 = np.array([[4.32718817e-01, 2.16789804e-05, 2.31465731e-11],
                    [4.31759394e-22, 1.02979145e-12, 1.09064040e+00],
                    [1.14580272e-11, 1.01088443e+00, 9.84217336e-10]])
    fdp2 = np.array([[0.32407312, 0.28118973, 0.23164959],
                    [0.18926917, 0.24622783, 0.32960461],
                    [0.2499131 , 0.25718975, 0.24536843]])
    fdp3 = np.array([[0.03332391, 0.0332761 , 0.03320984],
                    [0.03314437, 0.03323245, 0.03332958],
                    [0.0332373 , 0.03323259, 0.03322976]])
    
    l1,f1 =  func(xtrains, ytrains,xtests, 0.1)
    l2, f2 =  func(xtrains, ytrains,xtests, 1)
    l3, f3 =  func(xtrains, ytrains,xtests, 10)
    t1 =  ut.are_np_equal(l1, ytests_should1) and ut.are_np_equal(f1, fdp1)
    t2 =  ut.are_np_equal(l2, ytests_should1) and ut.are_np_equal(f2, fdp2)
    t3 =  ut.are_np_equal(l3, ytests_should2) and ut.are_np_equal(f3, fdp3)

    tests = {'test 1 fallo': t2,
             'test 2 fallo': t3,
             'test 3 fallo': t3,
              
             }
    test_res = ut.test_conditions_and_methods(tests)

    return (test_res)

@unknow_error
def test_parzen_exp(func):
    xx = np.array([[0,1], [1,1], [-1,1], [-1,0], [-0.9,0.15], [0.01,0.9], [0.9,0.9], [-0.99,0.99], [-0.8,-0.1]])
    yy =  np.array([0,0,0,0,1, 1,1,1,1])
    hs = [0.1,1,5]    
    res = ut.test_experimento_oneset(func,  shape_val=(len(hs), 3), 
                                    col_error = ['error de prueba(media)', 'error de prueba(desviación estandar)'],
                                    col_val=['acho de ventana', 'error de prueba(media)', 'error de prueba(desviación estandar)'],
                                    X = xx, Y=yy,
                                    hs = hs)
    return (res)


def part_1 ():
#cargamos la bd iris desde el dataset de sklearn
    from sklearn import datasets
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    GRADER = Grader("lab2_part2")
    GRADER.add_test("ejercicio1", Tester(test_muestras_por_clases))
    GRADER.add_test("ejercicio2", Tester(test_KNN_Clasificacion))
    GRADER.add_test("ejercicio3", Tester(test_train_test_split_fix)),
    GRADER.add_test("ejercicio4", Tester(test_exp))
    GRADER.add_test("ejercicio5", Tester(test_parzenClass))
    GRADER.add_test("ejercicio6", Tester(test_parzen_exp))
    return(GRADER, x,y)