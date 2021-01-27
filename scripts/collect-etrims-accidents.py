import project
from src.dataset import load_dataset
from src.config import config
from src import mongodb

user = config['mongodb']['user']
password = config['mongodb']['password']
host = config['mongodb']['host']
port = config['mongodb']['port']
auth_db = config['mongodb']['auth_db']

db = mongodb.connect(user, password, host, port, auth_db)

df = load_dataset()

