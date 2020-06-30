import pymongo


def connect(user, password, host, port, auth_db):
    # connect to the mongo database
    mongo_url = "mongodb://{}:{}@{}:{}/?authSource={}".format(
        user, password, host, port, auth_db)
    client = pymongo.MongoClient(mongo_url)
    db = client['data-class']
    print("collection names: {}".format(db.list_collection_names()))
    return db
