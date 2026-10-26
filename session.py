from dataclasses import asdict, dataclass, field
from enum import Enum
from uuid import uuid4
from models.reasons import (
    ClassificationStatus,
    Influence,
    ReasonCandidate,
    ReasonEvidence,
    ReasonRole,
    StructuredReason,
)
class SeparationContext(str, Enum):
    VOLUNTARY = "VOLUNTARY"
    INVOLUNTARY = "INVOLUNTARY"
    RETIREMENT = "RETIREMENT"
    CONTRACT_END = "CONTRACT_END"
    INTERNAL_TRANSFER = "INTERNAL_TRANSFER"
    OTHER = "OTHER"
    PREFER_NOT_TO_ANSWER = "PREFER_NOT_TO_ANSWER"
class ResponseDisposition(str, Enum):
    ANSWERED = "ANSWERED"
    SKIPPED = "SKIPPED"
    PREFER_NOT_TO_ANSWER = "PREFER_NOT_TO_ANSWER"
@dataclass(frozen=True)
class Response:
    response_id: str
    question_id: str
    text: str
    disposition: ResponseDisposition = ResponseDisposition.ANSWERED
    evidence: list[ReasonEvidence] = field(default_factory=list)


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    separation_context: SeparationContext | None = None
    responses: list[Response] = field(default_factory=list)
    candidates: list[ReasonCandidate] = field(default_factory=list)
    ranked_candidate_ids: list[str] = field(default_factory=list)
    ranking_response_ids: list[str] = field(default_factory=list)
    ranking_status: str = "NOT_APPLICABLE"
    clarification_attempts: dict[str, int] = field(default_factory=dict)
    reporting_state: str | None = None
    status: str = "IN_PROGRESS"
    def record_response(
        self,
        question_id: str,
        text: str,
        disposition: ResponseDisposition = ResponseDisposition.ANSWERED,
    ) -> Response:
        response = Response(f"R{len(self.responses) + 1}", question_id, text, disposition)
        self.responses.append(response)
        return response
    def record_evidence(
        self,
        response: Response,
        evidence: list[ReasonEvidence],
        related_response_ids: list[str] | None = None,
    ) -> None:
        response.evidence.extend(evidence)
        for item in evidence:
            candidate = next((c for c in self.candidates if c.key == item.key
                              and (item.category != "UNCLASSIFIED"
                                   or response.response_id in c.supporting_response_ids)), None)
            if candidate is None and item.supported:
                candidate = ReasonCandidate(
                    f"F{len(self.candidates) + 1}", item.category, item.subcategory, item.label,
                    status=(ClassificationStatus.UNCLASSIFIED if item.category == "UNCLASSIFIED"
                            else ClassificationStatus.ACTIVE),
                )
                self.candidates.append(candidate)
            if candidate is None:
                continue
            sources = [*(related_response_ids or []), response.response_id]
            for source in sources:
                if source not in candidate.source_response_ids:
                    candidate.source_response_ids.append(source)
            if item.supported:
                if response.response_id not in candidate.supporting_response_ids:
                    candidate.supporting_response_ids.append(response.response_id)
                candidate.add_influence(item.influence, response.response_id)
            for previous in self.responses:
                if any(e.key == item.key and not e.supported for e in previous.evidence):
                    if previous.response_id not in candidate.excluding_response_ids:
                        candidate.excluding_response_ids.append(previous.response_id)
                    if previous.response_id not in candidate.source_response_ids:
                        candidate.source_response_ids.append(previous.response_id)
            if candidate.supporting_response_ids and candidate.excluding_response_ids:
                candidate.status = ClassificationStatus.CONFLICTING
        if len(self.candidates) > 1 and self.ranking_status == "NOT_APPLICABLE":
            self.ranking_status = "PENDING"
    def mark_unclassified(self, source_response_ids: list[str]) -> None:
        response = next(r for r in self.responses if r.response_id == source_response_ids[0])
        self.record_evidence(
            response, [ReasonEvidence("UNCLASSIFIED", None, "unspecified reason")],
            source_response_ids[1:],
        )

    def rank_reasons(self, candidate_ids: list[str], response_ids: list[str]) -> None:
        if not 1 <= len(candidate_ids) <= 3 or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Participant ranking must contain one to three distinct factors.")
        available = {c.candidate_id for c in self.candidates}
        if not set(candidate_ids) <= available:
            raise ValueError("Participant ranking contains an unknown factor.")
        known_responses = {r.response_id for r in self.responses}
        if not response_ids or not set(response_ids) <= known_responses:
            raise ValueError("Participant ranking must reference participant responses.")
        self.ranked_candidate_ids = list(candidate_ids)
        self.ranking_response_ids = list(response_ids)
        self.ranking_status = "PARTICIPANT_RANKED"
    @property
    def reasons(self) -> list[StructuredReason]:
        by_id = {c.candidate_id: c for c in self.candidates}
        return [
            StructuredReason(
                category=by_id[candidate_id].category,
                subcategory=by_id[candidate_id].subcategory,
                rank=rank,
                role=ReasonRole.PRIMARY if rank == 1 else ReasonRole.CONTRIBUTING,
                influence=by_id[candidate_id].influence,
                status=by_id[candidate_id].status,
                source_response_ids=list(dict.fromkeys(
                    by_id[candidate_id].source_response_ids + self.ranking_response_ids
                )),
            )
            for rank, candidate_id in enumerate(self.ranked_candidate_ids, start=1)
        ]
    def to_dict(self) -> dict:
        summary = asdict(self)
        summary["reasons"] = [asdict(reason) for reason in self.reasons]
        return summary
