from os import listdir
from os.path import isfile, join

import fire

from src import TEST_FOLDER
from src.data.preprocessors.core import preprocess_data
# from src.data.preprocessors.matchers.core import preprocess_data
# from src.data.preprocessors.detectors.core import preprocess_data


def preprocess_test_data(file_name="."):
    if file_name == ".":
        test_files = [f for f in listdir(TEST_FOLDER) if isfile(join(TEST_FOLDER, f))]
    else:
        test_files = [file_name]
    for test_file_name in test_files:
        test_file_path = join(TEST_FOLDER, test_file_name)
        print(f'Processing {test_file_path}')
        preprocess_data(test_file_path)


if __name__ == "__main__":
    fire.Fire(preprocess_test_data)
