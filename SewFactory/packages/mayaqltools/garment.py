import os


class Garment(object):
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def to_filter_string(self):
        return self.type+"/"+self.name

    def to_rel_folder(self):
        return os.path.join(self.type, self.name)

    def to_abs_path(self, data_root):
        return os.path.join(data_root, self.type, self.name,)

    def to_spec_path(self, data_root):
        return os.path.join(data_root, self.type, self.name, "specification.json")
