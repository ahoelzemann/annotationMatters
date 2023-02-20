import platform
import os

if platform.system() != 'Windows':
    DELIMITER = '/'
    RESOURCES_ROOT = './local_files/'
    RESULTS_FOLDER = './results/'
else:
    DELIMITER = '\\'
    RESOURCES_ROOT = '.\\resources\\data\\'
    RESULTS_FOLDER = '.\\results\\'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)).split(DELIMITER + 'misc')[0]
WANDB_LOGGING = True
GLOBAL_HISTORY = []