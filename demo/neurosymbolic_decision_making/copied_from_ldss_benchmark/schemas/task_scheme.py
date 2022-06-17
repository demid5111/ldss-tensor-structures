# pylint: disable=invalid-name
"""
Schemas for validation of task creation REST request
"""
from typing import List, Dict, Optional

from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class CriterionDescription:
    criteriaID: str = Field(...,
                            title='ID of criterion')
    criteriaName: str = Field(...,
                              title='Name of criterion')
    qualitative: bool = Field(...,
                              title='Marker whether this criterion is qualitative')
    units: Optional[str] = Field(title='Units of measurement for the given criterion')
    benefit: Optional[bool] = Field(title='Flag denotes if this criteria is better is a value is bigger')


@dataclass
class AlternativeDescription:
    alternativeID: str = Field(...,
                               title='ID of alternative')
    alternativeName: str = Field(...,
                                 title='Name of alternative')
    abstractionLevelID: str = Field(...,
                                    title='ID of abstraction level to which alternative belongs')


@dataclass
class ScalesDescription:
    scaleID: str = Field(...,
                         title='ID of scale')
    scaleName: str = Field(...,
                           title='Name of scale')
    labels: List[str] = Field(...,
                              title='List of labels for the given linguistic scale')
    values: Optional[List[str]] = Field(
                              title='List of values for the given linguistic scale if it does not use fuzzy logic')


@dataclass
class AbstractionLevelDescription:
    abstractionLevelID: str = Field(...,
                                    title='ID of abstraction level')
    abstractionLevelName: str = Field(...,
                                      title='Name of abstraction level')


@dataclass
class ExpertDescription:
    expertID: str = Field(...,
                          title='ID of expert')
    expertName: str = Field(...,
                            title='Name of expert')
    competencies: List[str] = Field(...,
                                    title='List of personal competencies of the given expert')


@dataclass
class AlternativeAssessmentForSingleCriteriaDescription:
    criteriaID: str = Field(...,
                            title='ID of criterion')
    estimation: List[str] = Field(...,
                                  title='List of linguistic labels')
    scaleID: Optional[str] = Field(
        title='ID of linguistic scale that given estimation labels belong to')


@dataclass
class AlternativeAssessmentDescription:
    alternativeID: str = Field(...,
                               title='ID of alternative')
    criteria2Estimation: List[AlternativeAssessmentForSingleCriteriaDescription] = \
        Field(...,
              title='All assessments of given alternative by all criteria')


@dataclass
class ResultAlternativeAssessmentDescription:
    scaleID: str = Field(...,
                         title='ID of linguistic scale')
    alternativeID: str = Field(...,
                               title='ID of alternative')
    estimation: List[Dict[str, float]] = Field(...,
                                               title='Aggregated assessment')


@dataclass
class TaskDTOScheme:
    criteria: Dict[str, List[CriterionDescription]]
    criteriaWeightsPerGroup: Dict[str, List[float]]
    alternatives: List[AlternativeDescription]
    scales: List[ScalesDescription]
    abstractionLevels: List[AbstractionLevelDescription]
    abstractionLevelWeights: Dict[str, float]
    expertWeightsRule: Dict[str, float]
    experts: List[ExpertDescription]
    estimations: Dict[str, List[AlternativeAssessmentDescription]]
    expertWeights: Dict[str, float] = None

    class Config:
        arbitrary_types_allowed = True
