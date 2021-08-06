from os import listdir
from os.path import isfile, join

import fire

from src import TEST_FOLDER, logger
# from src.data.preprocessors.core import preprocess_data


from src.data.preprocessors.matchers.core import preprocess_data
# from src.data.preprocessors.detectors.core import preprocess_data


def preprocess_test_data(*file_names):
    if len(file_names) == 0:
        test_files = [f for f in listdir(TEST_FOLDER) if isfile(join(TEST_FOLDER, f))]
    elif isinstance(file_names, tuple):
        test_files = list(file_names)
    else:
        test_files = [file_names]
    for test_file_name in test_files:
        test_file_path = join(TEST_FOLDER, test_file_name)
        logger.info(f'Processing {test_file_name}')
        info, status = preprocess_data(test_file_path)
        logger.info(f'Finished {test_file_name}, info: {info}, status: {status}')
        logger.log_result(test_file_name, info, status, save=False)
    logger.save_results()


def main(*file_names, logs=None):
    if logs is not None:
        logger.setLevel(logs)
    preprocess_test_data(*file_names)


if __name__ == "__main__":
    fire.Fire(main)
