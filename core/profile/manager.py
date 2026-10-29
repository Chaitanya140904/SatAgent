import os
import yaml
import logging
from .schema import ProfileSchema

class ProfileManager:
    def __init__(self, yaml_path: str):
        self.logger = logging.getLogger("ProfileManager")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        if not os.path.exists(yaml_path):
            self.logger.critical(f"FATAL: Profile configuration missing at {yaml_path}")
            raise FileNotFoundError(f"Cannot find profile at {yaml_path}")

        self.logger.info(f"Ingesting orbital profile from {yaml_path}...")
        with open(yaml_path, 'r') as file:
            raw_data = yaml.safe_load(file)
            
        self.config = ProfileSchema(**raw_data)
        self.logger.info(f"Profile '{self.config.profile_name}' locked and validated.")
