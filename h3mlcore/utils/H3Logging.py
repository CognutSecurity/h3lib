'''
Misc functions for logging utils

Author: Huang Xiao
Group: Cognitive Security Technologies
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''

import os, yaml, logging 

def setup_logging(logging_config=None, level=logging.INFO):
    """Setup logging 
    
    Setup logging function using predefined config file. Log files will
    be saved in logging_root_dir according to the LEVEL, e.g., info.log
    
    Args:
      logging_root_dir (str): root folder where to save log files 
      logging_config (str): logging config file in YAML format    
    
    Returns:
      logger  
    
    """
    
    
    try:
        logging_config = yaml.safe_load(open(logging_config, 'r'))
        logging.config.dictConfig(logging_config)
    except IOError:
        logging.basicConfig(level=logging.INFO)
        logging.warning(
            "logging config file: %s does not exist. using default logging" % logging_config)
    finally:
        logger = logging.getLogger('default')
        logger.setLevel(level)
        return logger
