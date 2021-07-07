from os import listdir
from os.path import isfile, join

from src import TEST_FOLDER
from src.data.feature_matching_preprocessor import preprocess_data as feature_preprocess_data


def preprocess_test_data():
    test_files = [f for f in listdir(TEST_FOLDER) if isfile(join(TEST_FOLDER, f))]
    for test_file_name in test_files:
        test_file_path = join(TEST_FOLDER, test_file_name)
        print(test_file_path)
        # preprocess_data(test_file_path)
        feature_preprocess_data(test_file_path)


if __name__ == "__main__":
    preprocess_test_data()
