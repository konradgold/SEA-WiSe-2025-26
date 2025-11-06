import os
import redis
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

def connect_to_db(cfg):
    if cfg.REDIS_PROD.USE:
        logger.info("Connecting to production Redis database...")
        host = cfg.REDIS_PROD.HOST
        port = cfg.REDIS_PROD.PORT
        load_dotenv()
        password = os.getenv("REDIS_PROD_PASSWORD", None)
    else:
        host = cfg.REDIS_HOST
        port = cfg.REDIS_PORT
        password = None
    db = redis.Redis(host=host, port=port, password=password, decode_responses=True)
    assert db.ping(), "Could not connect to Redis database"
    return db