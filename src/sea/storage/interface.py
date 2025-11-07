from abc import ABC, abstractmethod
import json
from typing import Any, Generator
from sea.utils.manage_redis import connect_to_db
import logging
import os

logger = logging.getLogger(__name__)


class StorageInterface(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    def set(self, key, value):
        pass

    @abstractmethod
    def hset(self, key, value):
        pass

    @abstractmethod
    def get(self, key):
        pass

    @abstractmethod
    def hgetall(self, key):
        pass

    @abstractmethod
    def setnx(self, key, value) -> bool:
        pass

    @abstractmethod
    def delete(self, key) -> bool:
        pass

    @abstractmethod
    def dbsize(self) -> int:
        pass

    @abstractmethod
    def scan_iter(self, match=None, count=None) -> Generator[Any, Any, None]:
        pass

    @abstractmethod
    def mget(self, keys) -> Generator[Any, Any, None]:
        pass

    @abstractmethod
    def pipeline(self) -> Any:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def ping(self) -> bool:
        pass

    @abstractmethod
    def scan(self, cursor) -> tuple:
        pass

    @abstractmethod
    def execute(self) -> list:
        pass


class RedisStorage(StorageInterface):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.storage = connect_to_db(cfg)

    def hgetall(self, key):
        return self.storage.hgetall(key)
    
    def hset(self, key, value):
        return self.storage.hset(key, mapping=value)
    
    def get(self, key):
        return self.storage.get(key)
    
    def setnx(self, key, value):
        return self.storage.setnx(key, value)
    
    def delete(self, key):
        return self.storage.delete(key)
    
    def dbsize(self):
        return self.storage.dbsize()
    
    def scan_iter(self, match=None, count=None):
        for item in self.storage.scan_iter(match=match, count=count):
            yield item
    
    def mget(self, keys):
        return self.storage.mget(keys)
    
    def pipeline(self):
        return self.storage.pipeline()
    
    def close(self):
        return self.storage.close()
    
    def ping(self):
        return self.storage.ping()
    
    def scan(self, cursor):
        return self.storage.scan(cursor)

    def execute(self) -> list:
        return []

    def set(self, key, value):
        return self.storage.set(key, value)
    

class LocalStorage(StorageInterface):
    def __init__(self, cfg):
        super().__init__(cfg)
        try:
            if self.cfg.STORAGE.LOAD_DOCUMENTS:
                with open(self.cfg.STORAGE.PATH_DOCS, 'r') as f:
                    _storage_d = json.loads(f.read())
            else:
                _storage_d = {}
            with open(self.cfg.STORAGE.PATH_TOKENS, 'r') as f:
                _storage_t = json.loads(f.read())
            self._storage = {**_storage_d, **_storage_t}
        except FileNotFoundError:
            self._storage = {}
        self.add_counter = 0
        self.iterated = 0

    def save(self):
        storage_t = {k: v for k, v in self._storage.items() if k.startswith("token:")}
        storage_d = {k: v for k, v in self._storage.items() if k.startswith("D")}
        if self.cfg.STORAGE.KEEP_DOCUMENTS:
            with open(self.cfg.STORAGE.PATH_DOCS, 'w') as f:
                f.write(json.dumps(storage_d))
        with open(self.cfg.STORAGE.PATH_TOKENS, 'w') as f:
            f.write(json.dumps(storage_t))

    def set(self, key, value):
        if key not in self._storage:
            self.add_counter += 1
        self._storage[key] = value

    def hset(self, key, value):
        if key not in self._storage:
            self.add_counter += 1
        val = self._storage.get(key, {})
        val.update(value)
        self._storage[key] = val

    def get(self, key):
        return self._storage.get(key, None)

    def hgetall(self, key):
        return self._storage.get(key, {})

    def setnx(self, key, value):
        if key not in self._storage:
            self._storage[key] = value
            self.add_counter += 1
            return True
        return False

    def delete(self, key):
        if key in self._storage:
            del self._storage[key]
            self.add_counter += 1
            return True
        return False

    def dbsize(self):
        return len(self._storage.keys())


    def scan_iter(self, match=None, count=None):
        yielded = 0
        keys = list(self._storage.keys())[self.iterated:]
        for key in keys:
            self.iterated += 1
            if match is None or key.startswith(match):
                yield key
                yielded += 1
            if count is not None and yielded >= count:
                break
            if self.iterated >= self.dbsize():
                self.iterated = 0
                break

    def mget(self, keys):
        for key in keys:
            yield self._storage.get(key, None)

    def pipeline(self):
        return self

    def close(self):
        self.save()

    def ping(self):
        return True

    def scan(self, cursor, count=None):
        if count is None:
            count = self.cfg.STORAGE.CURSOR_SIZE if self.cfg.STORAGE.CURSOR_SIZE else 10
        keys = list(self._storage.keys())
        if cursor >= len(keys):
            return 0, []
        next_cursor = min(cursor + count, len(keys))
        return next_cursor, keys[cursor:next_cursor]

    def execute(self):
        counter = self.add_counter
        self.add_counter = 0
        return [counter]

def get_storage(cfg):
    if cfg.STORAGE.TYPE == "redis":
        logger.info("Using Redis storage backend.")
        return RedisStorage(cfg)
    elif cfg.STORAGE.TYPE == "filesystem":
        logger.info("Using local filesystem storage backend.")
        return LocalStorage(cfg)
    else:
        raise ValueError("Unknown storage type")