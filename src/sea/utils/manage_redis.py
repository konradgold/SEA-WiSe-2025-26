import os
import redis

def connect_to_db(cfg):
    if cfg.REDIS_PROD.USE:
        host = cfg.REDIS_PROD.HOST
        port = cfg.REDIS_PROD.PORT
        password = os.getenv("REDIS_PROD_PASSWORD", None)
    else:
        host = cfg.REDIS_HOST
        port = cfg.REDIS_PORT
        password = None
    return redis.Redis(host=host, port=port, password=password, decode_responses=True)