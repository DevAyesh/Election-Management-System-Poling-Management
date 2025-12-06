from django.db import models
from django.conf import settings
import pymongo
from datetime import datetime

# Connect to MongoDB
client = pymongo.MongoClient(settings.MONGO_URI)
db = client[settings.MONGO_DB_NAME]
voters_collection = db['voters']

class Voter:
    """
    Voter model wrapper for MongoDB.
    """
    def __init__(self, slin_nic, name, face_embedding, registered_at=None):
        self.slin_nic = slin_nic
        self.name = name
        self.face_embedding = face_embedding
        self.registered_at = registered_at or datetime.now()

    def save(self):
        voter_data = {
            "slin_nic": self.slin_nic,
            "name": self.name,
            "face_embedding": self.face_embedding,
            "registered_at": self.registered_at
        }
        voters_collection.update_one(
            {"slin_nic": self.slin_nic},
            {"$set": voter_data},
            upsert=True
        )

    @staticmethod
    def get_by_nic(slin_nic):
        data = voters_collection.find_one({"slin_nic": slin_nic})
        if data:
            return Voter(
                slin_nic=data['slin_nic'],
                name=data.get('name', ''),
                face_embedding=data.get('face_embedding', []),
                registered_at=data.get('registered_at')
            )
        return None
