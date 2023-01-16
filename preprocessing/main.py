from resolution_testing import preprocess_resolutions
from simple_model import test_resolutions

def preprocess_main():
    preprocess_resolutions()
    test_resolutions()

if __name__ == '__main__':
    preprocess_main()