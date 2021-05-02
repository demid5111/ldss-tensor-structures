class Model2Tuple:
    term_index: int
    linguistic_scale_size: int
    alpha: float
    weight: float

    def __init__(self, term_index: int, alpha: float, linguistic_scale_size: int = 5, weight: float = 0.5):
        self.term_index = term_index
        self.alpha = alpha
        self.linguistic_scale_size = linguistic_scale_size
        self.weight = weight

    def __str__(self):
        return f'(s^{self.linguistic_scale_size}_{self.term_index}, {self.alpha})'

    def to_number(self):
        tau = self.linguistic_scale_size - 1
        return (self.alpha + self.term_index) / tau

    @staticmethod
    def from_number(beta: float, linguistic_scale_size: int):
        tau = linguistic_scale_size - 1
        i = round(beta * tau)
        alpha = beta * tau - i
        return Model2Tuple(term_index=i, alpha=alpha, linguistic_scale_size=linguistic_scale_size)

    def __eq__(self, other):
        return (
                isinstance(other, Model2Tuple) and
                self.term_index == other.term_index and
                abs(self.alpha - other.alpha) < 0.1 and
                self.weight == other.weight and
                self.linguistic_scale_size == other.linguistic_scale_size
        )
