from resolution_testing import preprocess_resolutions
from simple_model import test_resolutions
from emailer import email

def preprocess_main():
    # preprocess_resolutions()

    for run in range(25):
        test_resolutions(run+1)
        email(run+1)

if __name__ == '__main__':
    preprocess_main()