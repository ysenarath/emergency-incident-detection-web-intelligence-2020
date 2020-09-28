import configparser
import os

config = configparser.ConfigParser()

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

config['DEFAULT'] = {
    'project_path': project_path,
    'data_path': os.path.join(project_path, 'output'),
    'data_file': 'waze_12_tmc+weather+etrims.csv',
}

config.read(os.path.join(project_path, 'config.ini'))
