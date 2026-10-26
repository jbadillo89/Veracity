from dataclasses import dataclass, field
from enum import Enum
class Influence(str, Enum):
    PUSH = "PUSH"
    PULL = "PULL"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"
class ClassificationStatus(str, Enum):
    ACTIVE = "ACTIVE"
    CONFLICTING = "CONFLICTING"
    UNCLASSIFIED = "UNCLASSIFIED"
class ReasonRole(str, Enum):
    PRIMARY = "PRIMARY"
    CONTRIBUTING = "CONTRIBUTING"
@dataclass(frozen=True)
class ReasonEvidence:
    category: str
    subcategory: str | None
    label: str
    supported: bool = True
    influence: Influence = Influence.UNKNOWN
    @property
    def key(self) -> tuple[str, str | None]:
        return self.category, self.subcategory
@dataclass
class ReasonCandidate:
    candidate_id: str
    category: str
    subcategory: str | None
    label: str
    status: ClassificationStatus = ClassificationStatus.ACTIVE
    influence: Influence = Influence.UNKNOWN
    source_response_ids: list[str] = field(default_factory=list)
    supporting_response_ids: list[str] = field(default_factory=list)
    excluding_response_ids: list[str] = field(default_factory=list)
    influence_response_ids: list[str] = field(default_factory=list)
    @property
    def key(self) -> tuple[str, str | None]:
        return self.category, self.subcategory
    def add_influence(self, influence: Influence, response_id: str) -> None:
        if influence is Influence.UNKNOWN:
            return
        if self.influence is Influence.UNKNOWN:
            self.influence = influence
        elif self.influence is not influence:
            self.influence = Influence.MIXED
        if response_id not in self.influence_response_ids:
            self.influence_response_ids.append(response_id)
        if response_id not in self.source_response_ids:
            self.source_response_ids.append(response_id)
@dataclass(frozen=True)
class StructuredReason:
    category: str
    subcategory: str | None
    rank: int
    role: ReasonRole
    influence: Influence
    status: ClassificationStatus
    source_response_ids: list[str]
