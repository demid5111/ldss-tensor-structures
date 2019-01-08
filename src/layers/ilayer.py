class ILayer:
    id = None

    def set_id(self, id):
        self.id = id

    def forward(self, *args, **kwargs):
        raise NotImplementedError
