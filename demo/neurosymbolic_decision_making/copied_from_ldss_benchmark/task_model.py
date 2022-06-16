from pathlib import Path

import numpy as np

from demo.neurosymbolic_decision_making.copied_from_ldss_benchmark.assessments.decision_matrix import DecisionMatrix
from demo.neurosymbolic_decision_making.copied_from_ldss_benchmark.converters import \
    _get_crisp_linguistic_assessment_from_json, _get_numeric_assessment_from_json
from demo.neurosymbolic_decision_making.copied_from_ldss_benchmark.models.alternatives import Alternatives
from demo.neurosymbolic_decision_making.copied_from_ldss_benchmark.models.criteria import Criteria
from demo.neurosymbolic_decision_making.copied_from_ldss_benchmark.models.scales import Scales
from demo.neurosymbolic_decision_making.copied_from_ldss_benchmark.schemas.task_scheme import TaskDTOScheme, AlternativeAssessmentDescription, \
    AlternativeAssessmentForSingleCriteriaDescription, ScalesDescription


class TaskModel:
    _dto: TaskDTOScheme
    _criteria: Criteria
    _scales: Scales
    _alternatives: Alternatives
    _decision_matrices: dict[str, DecisionMatrix]

    def __init__(self, dto: TaskDTOScheme, json_path: Path):
        self._dto = dto
        self._json_path = json_path
        self._criteria = Criteria(self._dto)
        self._scales = Scales(self._dto)
        self._alternatives = Alternatives(self._dto)
        self._decision_matrices = self.extract_raw_decision_matrices()

    @property
    def decision_matrices(self):
        return self._decision_matrices

    @property
    def json_path(self):
        return self._json_path

    @property
    def report(self):
        return {
            'num_alternatives': self.num_alternatives,
            'num_criteria': len(self._criteria),
            'num_experts': len(self._dto.experts),
            'num_criteria_groups': len(self._dto.criteria),
        }

    def extract_raw_decision_matrices(self):
        raw_matrices = self._load_raw_decision_matrices_from_dto()
        return self._unify_decision_matrices(raw_matrices)

    def _load_raw_decision_matrices_from_dto(self):
        alternatives_mapping = self._alternatives.alternative_to_index
        criteria_mapping = self._criteria.criterion_to_index

        num_alternatives = len(self._dto.alternatives)
        num_criteria = len(criteria_mapping)

        decision_matrices = {}
        for expert_id, expert_assessments in self._dto.estimations.items():
            expert_assessments: list[AlternativeAssessmentDescription]

            decision_matrix_np = np.empty((num_alternatives, num_criteria), dtype=object)

            for expert_assessment in expert_assessments:
                alt_idx = alternatives_mapping[expert_assessment.alternativeID]

                for criterion_assessment in expert_assessment.criteria2Estimation:
                    criterion_assessment: AlternativeAssessmentForSingleCriteriaDescription
                    criterion_idx = criteria_mapping[criterion_assessment.criteriaID]

                    decision_matrix_np[alt_idx][criterion_idx] = criterion_assessment

            decision_matrices[expert_id] = DecisionMatrix(decision_matrix_np)

        return decision_matrices

    def _unify_decision_matrices(self, raw_matrices: dict[str, DecisionMatrix]):
        all_matrices = {}
        for expert_id, decision_matrix in raw_matrices.items():
            unified_decision_matrix = np.empty_like(decision_matrix.get_raw(), dtype=float)
            for alternative_idx, criteria_values in enumerate(decision_matrix.get_raw()):
                for criterion_idx, criterion_assessment in enumerate(criteria_values):
                    if criterion_assessment.scaleID:  # is linguistic
                        scale: ScalesDescription = self._scales[criterion_assessment.scaleID]
                        if not scale.values:
                            return NotImplemented  # unable to work with fuzzy linguistic data
                        assert len(criterion_assessment.estimation) == 1, 'Crisp linguistic values cannot be fuzzy'
                        value = _get_crisp_linguistic_assessment_from_json(criterion_assessment.estimation[0], scale)
                    else:  # is numeric
                        assert len(criterion_assessment.estimation) == 1, 'Numeric values cannot be fuzzy'
                        value = _get_numeric_assessment_from_json(criterion_assessment.estimation[0])

                    unified_decision_matrix[alternative_idx][criterion_idx] = value

            all_matrices[expert_id] = DecisionMatrix(unified_decision_matrix)
        return all_matrices

    def get_criteria_weights(self):
        return self._criteria.criteria_weights

    def get_criteria_types(self):
        return self._criteria.criteria_types

    @property
    def num_alternatives(self):
        return len(self._alternatives)


class TaskModelFactory:
    def from_json(self, json_path: Path) -> TaskModel:
        with json_path.open(encoding='utf-8') as task_file:
            task_raw = task_file.read()
        # pylint: disable=no-member
        task_parsed = TaskDTOScheme.__pydantic_model__.parse_raw(task_raw)
        return TaskModel(task_parsed, json_path)
