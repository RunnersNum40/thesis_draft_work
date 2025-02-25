import warnings

from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm, trange

warnings.simplefilter("ignore", TqdmExperimentalWarning)
