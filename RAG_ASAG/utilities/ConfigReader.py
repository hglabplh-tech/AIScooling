import configparser


class ConfigReader:
    my_instance = None


    def __init__(self, config_file, config_section):
        self.config_section = config_section
        self.config_file = config_file
        self.config_parse = configparser.ConfigParser()
        self.config = self.config_parse.read(config_file)

    @staticmethod
    def myinstance (config_file, config_section):
        if ConfigReader.my_instance is None:
            ConfigReader.my_instance = ConfigReader(config_file, config_section)
        return ConfigReader.my_instance

    def read_val_float(self, value_key):
        return self.config_parse.getfloat(self.config_section, value_key)

    def read_val_int(self, value_key):
        return self.config_parse.getint(self.config_section, value_key)

    def read_val(self, value_key):
        return self.config_parse.get(self.config_section, value_key)

class Single:
    instance = None

