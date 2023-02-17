import sys
import numpy as np
from contextlib import closing

from six import StringIO

from gym import spaces
from gym import utils
from gym.envs.toy_text import discrete

from pprint import pprint


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "bridge_7": ["HHHHHHH", "GFSFFFG", "HHHHHHH"],
    "bridge_7_ns": ["HHHHHHH", "GFSFFFG", "HHHHFHH", "HHHHFFG"],
    "bridge_9": ["HHHHHHHHH", "GFFSFFFFG", "HHHHHHHHH"],
    "bridge_9_ns": ["HHHHHHHHH", "GFFSFFFFG", "HHHHHHHHH"],
    "bridge_15": ["HHHHHHHHHHHHHHH", "GFFFFFSFFFFFFFG", "HHHHHHHHHHHHHHH"],
    "bridge_15_ns": ["HHHHHHHHHHHHHHH", "GFFFFFSFFFFFFFG", "HHHHHHHHHHHHHHH"],
    "bridge_31": [
        "H" * 31,
        "G" + "F" * 13 + "S" + "F" * 15 + "G",
        "H" * 31,
    ],
}


class Bridge(discrete.DiscreteEnv):
    metadata = {"render.modes": ["human", "ansi"]}
    name = "Bridge"

    def __init__(self, desc=None, map_name="bridge_7"):
        self.map_name = map_name
        if desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b"S").astype("float64").ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        # li.append((1.0, s, 0, True))
                        pass
                    else:
                        if map_name == "bridge_7":
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b"GH"
                            if newletter == b"G":
                                # optimal goal
                                if newstate == 13:
                                    rew = 30
                                # suboptimal goal
                                elif newstate == 7:
                                    rew = 10
                            elif newletter == b"H":
                                # cliff
                                rew = -10
                            else:
                                # to find shortest path
                                rew = -1
                            li.append((1.0, newstate, rew, done))
                        elif map_name == "bridge_9":
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b"GH"
                            if newletter == b"G":
                                # optimal goal
                                if newstate == 17:
                                    rew = 30
                                # suboptimal goal
                                elif newstate == 9:
                                    rew = 10
                            elif newletter == b"H":
                                # cliff
                                rew = -10
                            else:
                                # to find shortest path
                                rew = -1
                            li.append((1.0, newstate, rew, done))
                        elif map_name == "bridge_15":
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b"GH"
                            if newletter == b"G":
                                # optimal goal
                                if newstate == 29:
                                    rew = 30
                                # suboptimalgoal
                                elif newstate == 15:
                                    rew = 10
                            elif newletter == b"H":
                                # cliff
                                rew = -10
                            else:
                                # to find shortest path
                                rew = -1
                            li.append((1.0, newstate, rew, done))
                        elif map_name == "bridge_31":
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b"GH"
                            if newletter == b"G":
                                # optimal goal
                                if newstate == 61:
                                    rew = 30
                                # suboptimalgoal
                                elif newstate == 31:
                                    rew = 10
                            elif newletter == b"H":
                                # cliff
                                rew = -10
                            else:
                                # to find shortest path
                                rew = -1
                            li.append((1.0, newstate, rew, done))

        super(Bridge, self).__init__(nS, nA, P, isd)

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Box(0, 1, shape=(self.nS,), dtype=np.float32)

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(
                "  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction])
            )
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()

    def reset(self):
        """
        Return a numpy array instead of an int.
        """
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        s_one_hot = np.zeros(self.nrow * self.ncol)
        s_one_hot[int(self.s)] = 1.0
        return s_one_hot

    def step(self, a):
        """
        State is a numpy array instead of an int.
        """
        if isinstance(a, np.ndarray):
            a = a.item()
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        s_one_hot = np.zeros(self.nrow * self.ncol)
        s_one_hot[int(s)] = 1.0
        return (s_one_hot, r, d, {"prob": p})


class BridgeNonStationary(discrete.DiscreteEnv):
    metadata = {"render.modes": ["human", "ansi"]}
    name = "Bridge Non-Stationary"

    def __init__(self, desc=None, map_name="bridge_7"):
        self.map_name = map_name
        if desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)
        self.ns_start = False

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b"S").astype("float64").ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        # li.append((1.0, s, 0, True))
                        pass
                    else:
                        if map_name == "bridge_7":
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b"GH"
                            if newletter == b"G":
                                # suboptimal goal
                                if newstate == 13:
                                    rew = 30
                                # goal
                                elif newstate == 7:
                                    rew = 10
                            elif newletter == b"H":
                                # cliff
                                rew = -10
                            else:
                                rew = -1
                            li.append((1.0, newstate, rew, done))

        super(BridgeNonStationary, self).__init__(nS, nA, P, isd)

    def trigger_non_stationarity(self, map_name="bridge_7_ns"):
        """
        Adds a new goal when the non-stationarity is triggered.
        Change the transition probability matrix, P.
        """
        desc = MAPS[map_name]

        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)
        self.ns_start = False

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b"S").astype("float64").ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        # li.append((1.0, s, 0, True))
                        pass
                    else:
                        if map_name == "bridge_7_ns":
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b"GH"
                            if newletter == b"G":
                                # suboptimal goal
                                if newstate == 13:
                                    rew = 30
                                # goal
                                elif newstate == 7:
                                    rew = 10
                                elif newstate == 27:
                                    rew = 40
                            elif newletter == b"H":
                                # cliff
                                rew = -10
                            else:
                                rew = -1
                            li.append((1.0, newstate, rew, done))

        super(BridgeNonStationary, self).__init__(nS, nA, P, isd)

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(
                "  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction])
            )
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()
