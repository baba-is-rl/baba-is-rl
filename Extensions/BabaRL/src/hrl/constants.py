import pyBaba

MAX_MAP_WIDTH = 36
MAX_MAP_HEIGHT = 24
MAX_RULES = 10
NUM_OBJECT_TYPES = max(int(t.value) for t in pyBaba.ObjectType.__members__.values()) + 1
PAD_INDEX = 0

MANAGER_SUBGOAL_FREQ_K = 25 

# Curriculum Constants
LEVELS = [
    "priming/lvl01a.txt",
    "priming/lvl01b.txt",
    "priming/lvl01c.txt",
    "priming/lvl02a.txt",
    "priming/lvl02b.txt",
    "priming/lvl02c.txt",
    "priming/lvl02d.txt",
    "priming/lvl03a.txt",
    "priming/lvl03b.txt",
    "priming/lvl03c.txt",
    "priming/lvl04a.txt",
    "priming/lvl04b.txt",
    "priming/lvl05a.txt",
    "priming/lvl05b.txt",
    "priming/lvl06a.txt",
    "priming/lvl06b.txt",
    "priming/lvl07a.txt",
    "priming/lvl07b.txt",
    "priming/lvl08a.txt",
    "priming/lvl08b.txt",
    "priming/lvl09a.txt",
    "priming/lvl09b.txt",
    "priming/lvl10.txt",
    "priming/lvl11a.txt",
    "priming/lvl11b.txt",
    "priming/lvl12.txt",
    "priming/lvl13.txt",
    "priming/lvl14.txt"
]
MAPS_DIR = "../../../../Resources/Maps"
SPRITES_PATH = "../../sprites"
CHECKPOINT_DIR_HRL_SUBGOAL = "./checkpoints_hrl_subgoal" 
LOG_DIR_HRL_SUBGOAL = "./logs_hrl_subgoal" 

CURRICULUM_THRESHOLD = 0.8 
CURRICULUM_WINDOW = 50    
