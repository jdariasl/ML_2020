import os

class Laboratory():

    def __init__ (self, data_paths, code_paths):

        self.data_path = data_paths
        self.code_path = code_paths
        self.repo_path = "https://raw.githubusercontent.com/jdariasl/ML_2020/labs/Labs/commons/utils/"
        print("lab configuration started")


    def download_github_code(self, path):
        filename = path.rsplit("/")[-1]
        os.system(f"wget {self.repo_path}{path} -O {filename}")

    def download_files(self):
        for d in self.data_path + self.code_path:
            self.download_github_code(d)

    def install_libraries(self):
        os.system("pip install gspread")
    
    def configure(self):
        print("installing libraries")
        self.install_libraries()
        print("downloading files")
        self.download_files()
        print("lab configured")

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

def configure_intro():
    data = []
    code = ["intro.py"]
    intro_lab = Laboratory(data, code)
    intro_lab.configure()
