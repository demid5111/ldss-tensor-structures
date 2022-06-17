from demo.neurosymbolic_decision_making.copied_from_ldss_benchmark.schemas.task_scheme import TaskDTOScheme, \
    CriterionDescription


class Criteria:
    _dto: TaskDTOScheme
    criterion_to_index: dict[str, int]
    criteria_full: dict[int, CriterionDescription]
    criteria_weights: list[float]
    criteria_types: list[float]  # contains True is criterion is benefit, False otherwise

    def __init__(self, task_dto: TaskDTOScheme):
        self._dto = task_dto
        self.criterion_to_index = {}
        self.criteria_full = {}

        # assert len(self._dto.criteriaWeightsPerGroup.keys()) == 1, 'Multiple criteria groups are not supported yet'
        self.criteria_weights = list(self._dto.criteriaWeightsPerGroup.values())[0]

        self.criteria_types = []

        criterion_index = 0
        for criteria_group in self._dto.criteria.keys():
            for criterion_info in self._dto.criteria[criteria_group]:
                criterion_info: CriterionDescription
                self.criterion_to_index[criterion_info.criteriaID] = criterion_index
                self.criteria_full[criterion_index] = criterion_info
                self.criteria_types.append(criterion_info.benefit)
                criterion_index += 1

    def __len__(self):
        return len(self.criterion_to_index)
