class ILayer:
    def __init__(self):
        self.id = None
        self.output = None

    def set_id(self, id):
        self.id = id

    def forward(self, *args, **kwargs):
        raise NotImplementedError
