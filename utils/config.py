"""Load project configurations from .env files.
Provides easy access to paths and credentials used in the project.
Meant to be used as an imported module.

If `config.py` is run on its own, it will create the appropriate
directories.

For information about the rationale behind decouple and this module,
see https://pypi.org/project/python-decouple/

Note that decouple mentions that it will help to ensure that
the project has "only one configuration module to rule all your instances."
This is achieved by putting all the configuration into the `.env` file.
You can have different sets of variables for difference instances, 
such as `.env.development` or `.env.production`. You would only
need to copy over the settings from one into `.env` to switch
over to the other configuration, for example.

"""

from decouple import config as env_config
from pathlib import Path
from pandas import to_datetime
from scheffer_quant import config as sq_config


# Absolute path to root directory of the project
BASE_DIR = Path(__file__).absolute().parent.parent

def if_relative_make_abs(path):
    path = Path(path)
    if path.is_absolute():
        abs_path = path.resolve()
    else:
        abs_path = (BASE_DIR / path).resolve()
    return abs_path


# fmt: off
## Other .env variables
#WRDS_USERNAME = env_config("WRDS_USERNAME", default="")
#NASDAQ_API_KEY = env_config("NASDAQ_API_KEY", default="")
START_DATE = env_config("START_DATE", default="1913-01-01", cast=to_datetime)
END_DATE = env_config("END_DATE", default="2024-12-31", cast=to_datetime)
USER = env_config("USER", default="")
PIPELINE_DEV_MODE = env_config("PIPELINE_DEV_MODE", default=True, cast=bool)
PIPELINE_THEME = env_config("PIPELINE_THEME", default="pipeline")

## Paths
DATA_DIR = if_relative_make_abs(env_config('DATA_DIR', default=Path('data'), cast=Path))
#RAW_DATA_DIR = Path(DATA_DIR / "raw")
#PROCESSED_DATA_DIR = Path(DATA_DIR / "processed")
#MANUAL_DATA_DIR = if_relative_make_abs(env_config('MANUAL_DATA_DIR', default=Path('data_manual'), cast=Path))
#LOG_DIR = if_relative_make_abs(env_config('LOG_DIR', default=Path('logs'), cast=Path))
OUTPUT_DIR = if_relative_make_abs(env_config('OUTPUT_DIR', default=Path('output'), cast=Path))
#PUBLISH_DIR = if_relative_make_abs(env_config('PUBLISH_DIR', default=Path('_output/publish'), cast=Path))

# Plot settings
PLOT_WIDTH = 12
PLOT_HEIGHT = 8

sq_config.Config.update(
    BASE_DIR=BASE_DIR,
    DATA_DIR=DATA_DIR,
    #MANUAL_DATA_DIR=MANUAL_DATA_DIR,
    #LOG_DIR=LOG_DIR,
    OUTPUT_DIR=OUTPUT_DIR,
    #PUBLISH_DIR=PUBLISH_DIR,
    #WRDS_USERNAME=WRDS_USERNAME,
    #NASDAQ_API_KEY=NASDAQ_API_KEY,
    START_DATE=START_DATE,
    END_DATE=END_DATE
)

if __name__ == "__main__":

    ## If they don't exist, create the _data and _output directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    #RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    #PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    #MANUAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    #PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    #LOG_DIR.mkdir(parents=True, exist_ok=True)
