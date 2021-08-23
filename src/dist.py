import numpy as np
import pickle
import random
import redis
import time

RESULTS_KEY = "results"
HOST = '10.100.10.10'
PORT = 6379
DB = 0
PASSWORD = ""


def serialize(x):
    """Return a pickled object."""
    return pickle.dumps(x)


def deserialize(x):
    """Return a depickled object."""
    return pickle.loads(x)


class Master(object):
    """Master that sends weights/coefficients to workers and waits for results from them."""
    def __init__(self, nodes):
        self.r = redis.Redis(host=HOST, port=PORT, db=DB, password=PASSWORD)
        self.run_id = 1
        self.nodes = nodes

        for key in self.r.scan_iter():
            print("deleting key", key)
            self.r.delete(key)

    def wait_for_results(self):
        """Wait for all workers to send fitnesses and seed to redis."""
        rewards = []
        seeds = []
        returnert = 0
        while returnert < self.nodes:
            _, res = self.r.blpop(RESULTS_KEY)
            rew, seed = deserialize(res)
            rewards.append(rew)
            seeds.append(seed)
            returnert += 1
            time.sleep(0.01)
        return rewards, seeds

    def send_weights(self, weights):
        """Send weights/coefficients to redis."""
        self.r.set("weights", serialize(weights))
        self.r.set("run_id", serialize(self.run_id))
        self.run_id += 1
    

class Worker(object):
    """Wait for weights/coefficients from master and return fitnesses and seed."""
    def __init__(self, run_id):
        self.r = redis.Redis(host=HOST, port=PORT, db=DB, password=PASSWORD)
        self.run_id = run_id
    
    def poll_weights(self):
        """Wait for new weights/coefficients in redis."""
        while True:
            new_run_id = deserialize(self.r.get("run_id"))
            time.sleep(0.1)
            if new_run_id != self.run_id:
                break
        
        self.run_id = new_run_id
        
        weights = deserialize(self.r.get("weights"))
        return weights

    def send_result(self, rew, seed):
        """Put fitnesses and seed in redis."""
        self.r.rpush(RESULTS_KEY, serialize((rew, seed)))
