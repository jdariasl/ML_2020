"""
Este archivo es generado automaticamente.

###### NO MODIFICAR #########

# cualquier alteración del archivo
# puede generar una mala calficacion o configuracion
# que puede repercutir negativamente en la 
# calificacion del laboratorio!!!!!

"""


import os

class Laboratory():

    def __init__ (self, data_paths, code_paths):

        self.data_path = [f"data/{data}" for data in data_paths]
        self.code_path = code_paths
        self.commons = ['imports.py']
        self.repo_path = "https://raw.githubusercontent.com/jdariasl/ML_2020/labs/Labs/commons/utils/"
        print("lab configuration started")


    def download_github_code(self, path):
        filename = path.rsplit("/")[-1]
        os.system(f"wget {self.repo_path}{path} -O {filename}")

    def download_files(self):
        for d in self.data_path + self.code_path+ self.commons:
            self.download_github_code(d)

    def install_libraries(self):
        os.system("pip install gspread")
    
    def configure(self):
        print("installing libraries")
        self.install_libraries()
        print("downloading files")
        self.download_files()
        print("lab configured")

class Grader():

    def __init__(self):
        self.tests = {}
        self.results = {}

    def add_test(self, name, test_to_add):
        self.tests[name] = test_to_add

    def run_test(self, name):
        results[name] = self.tests[name].run_test()

    def check_tests(self):
        if not( len(self.tests.keys()) == len(self.results.keys()) ):
            print("los tests estan incompletos! verifica el notebook")
            return None
    
        if (sum([res == 'nok' for res in self.results.values()]) >0 ):
            print("algunos de los test no estan ok. Verifica antes de enviar el formulario")
            return None

        print("Todo se ve ok. Asegurate de responder las preguntas en el formualario y envialo",
              "¡buen trabajo!")

class Tester():

    def __init__(self, name, func_for_testing, func_to_test):
        self.func_for_testing = func_for_testing
        self.f_to_test = f_to_test
        self.name = name

    def run_test(self):
        res = self.func_for_testing(self.func_to_test)
        if (res):
            return ("ok")
        else:
            return("nok")

def is_func(f):
    import types
    res = isinstance(f, types.FunctionType)
    if not (res):
        print("Revisa tu funcion!!!", 
             "parace ser que no creaste una funcion," , 
             "presta atencion a las instrucciones, o pregunta al profesor!")
    return (res)


### configuration for each lab
def configure_intro():
    data = ['bank.csv']
    code = ["intro.py"]
    intro_lab_object = Laboratory(data, code)
    intro_lab_object.configure()
    print(f"execute import {code} from *")
    return(intro_lab_object)
