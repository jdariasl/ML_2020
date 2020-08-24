"""
Este archivo es generado automaticamente.

###### NO MODIFICAR #########

# cualquier alteración del archivo
# puede generar una mala calficacion o configuracion
# que puede repercutir negativamente en la 
# calificacion del laboratorio!!!!!

"""

from imports import *

# for examples of exercise
matriz_a_for =  np.array(range(100)).reshape((10,10))
matriz_b_for = 2*np.ones((1,10))

def intro_grade_example(func):
    print("ejecutando test")
    if is_func(func):
        return None
    test1 = np.random.rand(3,3)
    test2 = np.random.rand(3,3)
    res = np.dot(test1, test1)*2
    to_test = func(test1, test2)
    if (res == to_test):
        return(True)
    else:
        return (False)


GRADER_INTRO_LAB = Grader()
GRADER_INTRO_LAB.add_test("ejercicio1", intro_grade_example)
    

        
