import pyBaba

MAX_MAP_WIDTH = 36
MAX_MAP_HEIGHT = 24
MAX_RULES = 10
NUM_OBJECT_TYPES = max(int(t.value) for t in pyBaba.ObjectType.__members__.values()) + 1
PAD_INDEX = 0

MANAGER_UPDATE_FREQ_K = 10  # Manager chooses a new option every K steps
NUM_OPTIONS = 4           # Number of different options/goals the Manager can choose