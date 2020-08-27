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

# for examples of exercise
matriz_a_for =  np.array(range(100)).reshape((10,10))
matriz_b_for = 2*np.ones((1,10))

def genarete_data():
    xtrain1 = np.random.rand(3,3)
    xtrain2 = np.random.rand(3,3)
    xtrain3 = np.random.rand(3,4)
    yt1 = np.ones(3)
    xtrain4 = np.random.rand(10,2)
    yt2 = np.ones(10)
    return(xtrain1, xtrain2, xtrain3, yt1, xtrain4, yt2)


@unknow_error
def intro_grade_example3(func):
    ut = Utils()
    xtrain1, xtrain2, xtrain3, yt1, xtrain4, yt2 = genarete_data()
    df1 = func(xtrain1, yt1, range(5)) 
    df2 = func(xtrain4, yt2, range(10)) 
    tests = {'Debes retornar los parametros sugeridos':  df1.shape[0] == 5,
             'Recuerda que la funcion debe recibir las variables del modelo como parametros':  df2.shape[0] == 10,
             'Debes retornar dos columnas!' : ut.test_columns_len(df1, 2) and ut.test_columns_len(df2, 2)}
    test_res = ut.test_conditions_and_methods(tests)
    return (test_res)
    

@unknow_error
def intro_grade_example2(func):
    xtrain1, xtrain2, xtrain3, yt1, xtrain4, yt2 = genarete_data()
    w1, e1 = func(xtrain1, yt1, 0) 
    w2, e2 = func(xtrain2, yt1, 1)
    w3, e3 = func(xtrain3, yt1, 2)
    w4, e4 = func(xtrain4, yt2, 3)
    one_test =( w1 != w2).all()
    all_tests = ((w1 != w2 ).all() and
                 w1.shape[0] == w2.shape[0] and
                 w3.shape[0] == 4 and
                 (e1 != e2).all())
    err_num =( type(e2).__module__  == np.__name__) and ( type(e4).__module__  == np.__name__)
    if (not (one_test)):
        print("recuerda que debe inicializar aleatorimanete")
        return (False)
    elif not((all_tests)):
        print("unos de los tests no pasaron, revisa tu implementacion")
        return (False)
    elif (not(err_num)):
        print("el valor de error no es el correcto")
        return (False)
    else:
        print("Todos los test pasaron!")
        return (True)

@unknow_error
def intro_grade_example(func):
    test1 = np.random.rand(3,3)
    test2 = np.random.rand(3,3)
    res = np.dot(test1, test2)
    to_test = func(test1, test2)
    if ((res == to_test).all()):
        return(True)
    else:
        print("Lo siento. revisa tu implementación.")
        return (False)

GRADER_INTRO_LAB = Grader("intro")
test1 = Tester(intro_grade_example)
GRADER_INTRO_LAB.add_test("ejercicio1", test1)
test2 = Tester(intro_grade_example2)
GRADER_INTRO_LAB.add_test("ejercicio2", test2)
test3 = Tester(intro_grade_example3)
GRADER_INTRO_LAB.add_test("ejercicio3", test3)
    

        
