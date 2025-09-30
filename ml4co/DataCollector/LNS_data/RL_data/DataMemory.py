import numpy as np

class RingBuffer(object):
    def __init__(self, maxlen, shape=None):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.shape = shape

        if shape is not None:
            self.data = np.zeros((maxlen,) + shape, dtype=np.float32)
        else:
            self.data = [None] * maxlen

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError("Index out of bounds")
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return [self[i] for i in idxs]

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        else:
            self.start = (self.start + 1) % self.maxlen
        self.data[(self.start + self.length - 1) % self.maxlen] = v

def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, action_shape, observation_shapes):
        self.limit = limit

        self.observations0 = {
            key: RingBuffer(limit, shape=shape) for key, shape in observation_shapes.items()
        }
        self.observations1 = {
            key: RingBuffer(limit, shape=shape) for key, shape in observation_shapes.items()
        }

        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.actions_next = RingBuffer(limit, shape=action_shape)
        self.instance = RingBuffer(limit, shape=(1,))



    def sample(self, batch_size):
        
        batch_idxs = np.random.choice(self.nb_entries, batch_size, replace=False)

        obs0_batch = {key: self.observations0[key].get_batch(batch_idxs) for key in self.observations0}
        obs1_batch = {key: self.observations1[key].get_batch(batch_idxs) for key in self.observations1}

        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        next_action_batch = self.actions_next.get_batch(batch_idxs)
        instance_batch = self.instance.get_batch(batch_idxs)

        result = {
            'obs0': {k: array_min2d(v) for k, v in obs0_batch.items()},
            'obs1': {k: array_min2d(v) for k, v in obs1_batch.items()},
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'next_actions': array_min2d(next_action_batch),
            'ins_ind': array_min2d(instance_batch),
        }
        return result

    
    def append(self, obs0, action, reward, obs1, next_action, ins, training=True):
        if not training:
            return

        for key in self.observations0:
            self.observations0[key].append(obs0[key].cpu())
            self.observations1[key].append(obs1[key].cpu())

        self.actions.append(action)
        self.rewards.append(reward)
        self.actions_next.append(next_action)
        self.instance.append(ins)

    @property
    def nb_entries(self):
        return len(self.actions)  # 或者 len(self.rewards) 等
