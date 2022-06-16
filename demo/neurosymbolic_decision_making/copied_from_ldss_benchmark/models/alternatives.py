from demo.neurosymbolic_decision_making.copied_from_ldss_benchmark.schemas.task_scheme import TaskDTOScheme, \
    AlternativeDescription


class Alternatives:
    alternative_to_index: dict[str, int]
    _dto: TaskDTOScheme

    def __init__(self, task_dto: TaskDTOScheme):
        self._dto = task_dto
        self.alternative_to_index = {}

        for alt_idx, alt_info in enumerate(self._dto.alternatives):
            alt_info: AlternativeDescription
            self.alternative_to_index[alt_info.alternativeID] = alt_idx

    def __len__(self):
        return len(self.alternative_to_index)
