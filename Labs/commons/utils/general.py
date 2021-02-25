"""
Este archivo es generado automaticamente.

###### NO MODIFICAR #########

# cualquier alteración del archivo
# puede generar una mala calificación o configuracion
# que puede repercutir negativamente en la 
# calificación del laboratorio!!!!!

###### NO MODIFICAR #########
"""

import os
import functools
import traceback
import time
import pandas as pd
import numpy as np
import sklearn
import inspect

class Laboratory():

    def __init__ (self, data_paths, code_paths):

        self.data_path = [f"data/{data}" for data in data_paths]
        self.code_path = code_paths
        self.commons = ['imports.py']
        self.repo_path = "https://raw.githubusercontent.com/jdariasl/ML_2020/master/Labs/commons/utils/"
        print("lab configuration started")


    def download_github_code(self, path):
        filename = path.rsplit("/")[-1]
        os.system(f"wget {self.repo_path}{path} -O {filename}")

    def download_files(self):
        for d in self.data_path + self.code_path+ self.commons:
            self.download_github_code(d)

    def install_libraries(self):
        os.system("pip install gspread ")
        # for avoid a bug qith seaborn
        os.system("pip install matplotlib<3.3.1")
        os.system("pip install scikit-learn==0.23")

    def configure(self):
        print("installing libraries")
        self.install_libraries()
        print("downloading files")
        self.download_files()
        print("lab configured")

class Grader():

    def __init__(self, lab_name, num_questions = 4):
        self.tests = {}
        self.results = {}
        self.lab_name = lab_name
        self.num_questions = num_questions

    def add_test(self, name, test_to_add):
        self.tests[name] = test_to_add

    def run_test(self, name, f_to_test):
        if name not in self.tests:
            print("verifica el orden de ejecucion de tu notebook!",  
                 "parece que no has ejecutado en el orden correcto",
                 "si tienes una duda, consultalo con el profesor, preguntando por",
                 f"FALLA en '{self.lab_name}-{name}'")
            return None

        self.results[name] = self.tests[name].run_test(f_to_test)

    def check_tests(self):
        if not( len(self.tests.keys()) == len(self.results.keys()) ):
            print("los tests estan incompletos! verifica el notebook")
            return None
    
        if (sum([res == 'nok' for res in self.results.values()]) >0 ):
            print("algunos de los test no estan ok. Verifica antes de enviar el formulario")
            return None

        print("Todo se ve ok. Asegurate de responder las preguntas abiertas y envia e archivo al formulario",
              "¡buen trabajo!")
    
    def grade(self, worksheet = None, int_range = None, students = None,  open_asnwers = None):
        
        if worksheet is None:
            print("uso del docente")
            return(None)
    
        print("resgister students")
        # register the students
        int_list = worksheet.range(*int_range)
        int_list[0].value =  ''.join([s for s in  students[0].strip() if s.isdigit()]) 
        int_list[1].value =  ''.join([s for s in  students[1].strip() if s.isdigit()]) 
        worksheet.update_cells(int_list)

        # register code_excercises
        total_q_code_ex = len(open_asnwers)*2 + len(self.results)
        answer_range = (int_range[0],int_range[1]+2, int_range[2], int_range[3]+2+total_q_code_ex)
        ans_list = worksheet.range(*answer_range)

        print("resgister code exercises")
        for n,(k,v) in enumerate (self.tests.items()):
            ans_list[n].value = f'{k}:{self.results[k]}' if k in self.results else f'{k}:nok'
        print("resgister open exercises")
        # add a blank cell after each answer
        idx = 1
        for nn, o_ans in enumerate (open_asnwers):
            if nn == 0:
                ans_list[n+nn+1].value = o_ans if o_ans.strip() !='' else 'no respuesta'
            else:
                ans_list[n+nn+1+idx].value = o_ans if o_ans.strip() !='' else 'no respuesta'
                idx = idx+1

        worksheet.update_cells(ans_list)
        
        

        

class Tester():

    def __init__(self, func_for_testing):
        self.func_for_testing = func_for_testing
    
    def run_test(self, func_to_test):
        res = self.func_for_testing(func_to_test)
        if (res):
            print("TEST EXITOSO!")
            return ("ok")
        else:
            print("FALLIDO. revisa tu funcion. Sigue las instrucciones del notebook. Si tienes alguna duda pregunta!")
            return("nok")

### Utils
class Utils():

    def __init__(self):
            pass

    def is_func_tester(self, f):
        import types
        res = isinstance(f, types.FunctionType)
        if not (res):
            print("....¡Revisa tu codigo!....", 
                "parace ser que no creaste una funcion," , 
                "presta atención a las instrucciones, o pregunta al profesor")
        return (res)

    def is_dataframe_tester(self, df):
        test = isinstance(df, pd.DataFrame)
        if not (test):
            print("recuerda que debes devolver un pandas DataFrame")
            return(False)
        else:
            return(True)
        
    def test_columns_len(self, df, test_len):
        return (len(df.columns) == test_len)

    def test_conditions_and_methods(self, test_msgs):
        """ test the values of dict and return the msj (key) if the test is false """
        for msj,test in test_msgs.items():
            if not(test):
                print(msj)
                return(False)
        return (True)

    def are_np_equal(self, x1,x2, neg = False):
        """ compare if 2 numpy arrays are equal """
        try: 
            #comparison = x1 == x2
            #equal_arrays = comparison.all()
            equal_arrays = np.allclose(x1,x2) if not neg else  not(np.allclose(x1,x2))
            if not(equal_arrays):
                print("un test fallido por que estos dos arrays no son iguales \n", x1,"\n--\n", x2)
                #print("un test fallido revisa tu funcion.")
            return (equal_arrays)
        except AttributeError:
            print("un test fallido por que estos dos arrays no son iguales \n",x1,"\n--\n", x2)
            #print("un test fallido revisa tu funcion.")
            return (False)
        except Exception as e:
            raise e

    def test_experimento(self, func, xtrain, ytrain, shape_val=None, col_val= None,  **kwargs):

        df1 = func(xtrain, xtrain, ytrain, ytrain, **kwargs)
        shape_test = df1.shape == shape_val
        cols_test = list(df1.columns) == col_val

        tests = {'Recuerda la funcion debe retornar un dataframe': self.is_dataframe_tester(df1),
                'Revisa tu implementacion. \n el df no tiene los experimentos requeridos. \n evita dejar codigo estatico ': shape_test,
                'Revisa tu implementación\n el df no tiene las columnas requeridas': cols_test}
                
        test_res = self.test_conditions_and_methods(tests)
        return (test_res)

    def test_experimento_oneset(self, func, col_error=None, shape_val=None, col_val= None,   return_df = False, **kwargs):

        df1 = func (**kwargs)
        shape_test = df1.shape == shape_val
        #print( df1.shape, shape_val)
        cols_test = list(df1.columns) == col_val
        error_t = True
        for c_e in col_error:
            error_t = (df1[c_e].nunique() > 1) and (error_t)
        
        #print(df1)
        if len(col_error)>1:
            error_t = error_t and not(df1[col_error].eq(df1[col_error].iloc[:, 0], axis=0).all().all())
            #print(df1[col_error].eq(df1[col_error].iloc[:, 0], axis=0).all().all())
        else:
            error_t = error_t

        tests = {'Recuerda la funcion debe retornar un dataframe': self.is_dataframe_tester(df1),
                'Revisa tu implementacion. \n el df no tiene los experimentos requeridos. \n evita dejar codigo estatico ': shape_test,
                'Revisa tu implementación\n el df no tiene las columnas requeridas': cols_test,
                 'El error es constante,o no se están retornando las columnas adecuadas revisa tu implementacion' : error_t }
                
        test_res = self.test_conditions_and_methods(tests)
        if return_df:
            return(test_res, df1)
        else:
            return (test_res)

    def test_experimento_train_test(self, func, xtrain, ytrain, xtest, ytest, shape_val=None, col_val= None, col_error=None, **kwargs):

        df1 = func(xtrain, xtest, ytrain, ytest, **kwargs)
        shape_test = df1.shape == shape_val
        cols_test = list(df1.columns) == col_val

        t1 = (df1['error_entreamiento'] == df1['error_prueba']).sum() != df1.shape[0]
        error_t = True
        for c_e in col_error:
            error_t = (df1[c_e].nunique() > 1) and (error_t)

        tests = {'Recuerda la funcion debe retornar un dataframe': self.is_dataframe_tester(df1),
                'Revisa tu implementacion. \n el df no tiene los experimentos requeridos. \n evita dejar codigo estatico ': shape_test,
                'Revisa tu implementación\n el df no tiene las columnas requeridas': cols_test,
                'Recuerda que debes retornar el error de entrenamiento y de pruebas': t1,
                'El error es constante,o no se están retornando las columnas adecuadas revisa tu implementacion' : error_t }
                
        test_res = self.test_conditions_and_methods(tests)
        return (test_res)

    def get_data_from_scatter(self,ax):
        collects= ax.collections
        datax = []
        for coll in collects:
            datax.append(coll.get_offsets().data)
        
        return (np.vstack(datax))

    def get_org_data(self,x,y):
        datax = []
        for coll in np.unique(y):
            datax.append(x[np.ravel(y) == coll, :])
        return (np.vstack(datax))

    #generate linear separable datasets
    def get_linear_separable_dataset(self,random_state = 10, ext = True):
        from sklearn.datasets import make_classification, make_gaussian_quantiles
        X, Y  = make_classification(n_samples=50, n_features= 2, n_informative=2, n_redundant=0,
                            n_clusters_per_class=1, class_sep = 1, random_state= random_state)
        if ext:
            unos = np.array([np.ones(X.shape[0])])
            X = np.concatenate((unos.T, X), axis=1)
            X = X.reshape(X.shape[0], X.shape[1])
            Y = Y.reshape(np.size(Y), 1)
        return (X,Y)

    def get_nolinear_separable_dataset(self, random_state = 10, ext = True):
        from sklearn.datasets import make_gaussian_quantiles
        X, Y  = make_gaussian_quantiles(n_samples=50, n_features= 2, n_classes = 2, random_state= random_state)
        if ext:
            unos = np.array([np.ones(X.shape[0])])
            X = np.concatenate((unos.T, X), axis=1)
            X = X.reshape(X.shape[0], X.shape[1])
            Y = Y.reshape(np.size(Y), 1)
        return (X,Y)

    def is_inc_dec(self, error, increasing = True):
        e=np.ravel(error)
        return ( (e[1:] >= e[:-1]).all() if increasing else (e[1:] <= e[:-1]).all()  )

    def get_source_safe(self,func):
        codes = inspect.getsource(func)
        # remove docs
        codes = codes.split('\n')
        codes = "".join([c for c in codes if not(c.startswith("#")) ])
        codes = codes.replace(' ', '')
        return(codes)

    
    def check_code (self, code_to_look, func, msg = "*** recordar usar la libreria de sklearn y llamar explicitamente el/los parametro(s) correcto(s)! ***", debug=False):
        
        tests = [c not in self.get_source_safe(func) for c in code_to_look]
        if np.any(tests):
            print (msg)
            if debug:
                print(tests)
            return (False)
        else:
            return(True)



### decorators
def unknow_error(func):
    """decorates functions to return the original error"""
    @functools.wraps(func)
    def wrapper (*args, **kwargs):
        ut = Utils()
        if not ut.is_func_tester(func):
            return False
        try:
            return (func(*args, **kwargs))
        except Exception as e: 
            print("...error inesperado....\n ", "es muy probable que tengas un error de sintaxis \n", 
            "...puedas que tengas una variable definida erroneamente dentro de la funcion... \n"
            "...este es el stack retornado...\n .... \n")
            traceback.print_exc()
            return False
    return (wrapper)

### -------------------------
### configuration for each lab
###---------------------------
## intro
def configure_intro():
    data = ['bank.csv']
    code = ["intro.py"]
    intro_lab_object = Laboratory(data, code)
    intro_lab_object.configure()

def configure_lab1_p1():
    data = ['AirQuality.data']
    code = ["lab1.py"]
    intro_lab_object = Laboratory(data, code)
    intro_lab_object.configure()

def configure_lab1_p2():
    data = ['DatosClases.mat']
    code = ["lab1.py"]
    intro_lab_object = Laboratory(data, code)
    intro_lab_object.configure()

def configure_lab2():
    data = []
    code = ["lab2.py"]
    intro_lab_object = Laboratory(data, code)
    intro_lab_object.configure()

def configure_lab3():
    data = []
    code = ["lab3.py"]
    intro_lab_object = Laboratory(data, code)
    intro_lab_object.configure()


def configure_lab4():
    data = ["AirQuality.data"]
    code = ["lab4.py"]
    intro_lab_object = Laboratory(data, code)
    intro_lab_object.configure()


def configure_lab5_1():
    data = ["international-airline-passengers.csv"]
    code = ["lab5.py"]
    intro_lab_object = Laboratory(data, code)
    intro_lab_object.configure()


def configure_lab5_2():
    data = ["AirQuality.data"]
    code = ["lab5.py"]
    intro_lab_object = Laboratory(data, code)
    intro_lab_object.configure()

def configure_lab6():
    data = ['DB_Fetal_Cardiotocograms.txt']
    code = ["lab6.py"]
    intro_lab_object = Laboratory(data, code)
    intro_lab_object.configure()