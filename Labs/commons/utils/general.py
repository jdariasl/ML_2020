import os

class Laboratory():

    def __init__ (self, data_path, code_path):

        self.data_path = data_path
        self.code_path = code_path
        self.repo_path = "https://github.com/jdariasl/ML_2020/blob/labs/Labs/commons/utils/"
        print("lab configured")


    def download_github_code(self, path):
        filename = path.rsplit("/")[-1]
        os.system("wget https://raw.githubusercontent.com/hse-aml/natural-language-processing/master/{} -O {}".format(path, filename))

    def download_files(self):
        pass

    def install_libraries(self):
        os.system("pip install gspread")
    
    def configure(self):
        print("installing libraries")
        self.install_libraries()
        print("downloading files")
        self.download_files()

class Grader ():

    def __init__(self):
        self.tests = []

    def add_test(self, test_to_add):
        self.tests.append(test_to_add)

    def get_tests(self):
        results = []
        for t in tests :
            res = t.run_test()
            results.append(t)


class Test():

    def __init__(self, name, func_to_test, expected_result):
        self.func_to_test = func_to_test
        self.expected = expected_result
        self.name = name

    def run_test(self):
        res = self.funct_to_test()
        if (res == self.expected):
            return ("ok")
        else:
            return("nok")
        