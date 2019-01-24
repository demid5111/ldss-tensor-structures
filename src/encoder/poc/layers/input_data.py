from src.encoder.poc.layers.ilayer import ILayer


class InputData(ILayer):
    def __init__(self):
        super().__init__()
        self.input_data = None
        self.output = None

    def forward(self, *args, **kwargs):
        self.output = self.input_data
        return self.output
