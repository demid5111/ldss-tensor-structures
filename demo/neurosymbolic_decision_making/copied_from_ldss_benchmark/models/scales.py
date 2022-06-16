from demo.neurosymbolic_decision_making.copied_from_ldss_benchmark.schemas.task_scheme import TaskDTOScheme, ScalesDescription


class Scales:
    _dto: TaskDTOScheme
    _scales_mapping: dict[str, ScalesDescription]

    def __init__(self, task_dto: TaskDTOScheme):
        self._dto = task_dto
        self._scales_mapping = self._extract_scales()

    def __getitem__(self, item):
        return self._scales_mapping[item]

    def _extract_scales(self):
        return {scale.scaleID: scale for scale in self._dto.scales}
