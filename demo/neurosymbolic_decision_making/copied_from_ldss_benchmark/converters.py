from demo.neurosymbolic_decision_making.copied_from_ldss_benchmark.schemas.task_scheme import ScalesDescription


def _get_crisp_linguistic_assessment_from_json(value: str, scale: ScalesDescription):
    value_idx = scale.labels.index(value)
    return int(scale.values[value_idx])


def _get_numeric_assessment_from_json(json_value):
    return float(json_value)
