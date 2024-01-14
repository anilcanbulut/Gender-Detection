import yaml

class Config():
    def __init__(self, config_path) -> None:
        self.config_path = config_path

    def read_config(self):
        assert ".yaml" in self.config_path, f"{self.config_path} is not a '.yaml' file path!"
        print("Loading the config the file")
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        return config
    
    def find_simple_keys(self, yaml_data, parent_key=''):
        simple_keys = {}
        for key, value in yaml_data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if not isinstance(value, dict):  # Check if the value is not a dictionary
                simple_key = full_key.split('.')[-1]
                simple_keys[simple_key] = value
            else:
                child_keys = self.find_simple_keys(value, full_key)
                simple_keys.update(child_keys)
        return simple_keys