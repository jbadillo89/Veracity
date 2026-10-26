import re
from dataclasses import dataclass, field
from models.reasons import ClassificationStatus, Influence, ReasonEvidence
from nlp.intents import normalize, normalize_label
from nlp.taxonomy import Taxonomy, load_knowledge


@dataclass
class ClassificationResult:
    evidence: list[ReasonEvidence] = field(default_factory=list)

    @property
    def supported(self) -> list[ReasonEvidence]:
        return [item for item in self.evidence if item.supported]

    @property
    def excluded(self) -> list[ReasonEvidence]:
        return [item for item in self.evidence if not item.supported]

    @property
    def status(self) -> ClassificationStatus:
        return ClassificationStatus.ACTIVE if self.supported else ClassificationStatus.UNCLASSIFIED


class Processor:
    def __init__(self, taxonomy: Taxonomy | None = None) -> None:
        self.taxonomy = taxonomy or Taxonomy()
        data = load_knowledge("classification_rules.json")
        self.boundaries = re.compile(data["clause_boundaries"])
        self.exclusions = self._compile(data["exclusion_patterns"])
        self.uncertainty = self._compile(data["uncertainty_patterns"])
        self.negations = self._compile(data["negation_patterns"])
        self.pull_patterns = self._compile(data["pull_patterns"])
        self.rules = []
        for item in data["rules"]:
            self.taxonomy.require_valid(item["category"], item["subcategory"])
            self.rules.append({
                **item,
                "topics": self._compile(item["topics"]),
                "patterns": self._compile(item["patterns"]),
                "explicit_labels": {normalize_label(label) for label in item["explicit_labels"]},
            })
        self.broad_factors = data["broad_factors"]
        for item in self.broad_factors:
            self.taxonomy.require_valid(item["category"], item["subcategory"])
        self.influence_answers = {
            normalize_label(alias): Influence(influence)
            for influence, aliases in data["influence_answers"].items() for alias in aliases
        }
        self.clarifications = data.get("clarifications", [])
        for clarification in self.clarifications:
            for interpretation in clarification["interpretations"]:
                self.taxonomy.require_valid(interpretation["category"], interpretation["subcategory"])

    @staticmethod
    def _compile(patterns: list[str]) -> list[re.Pattern]:
        return [re.compile(pattern) for pattern in patterns]
    @staticmethod
    def _matches(patterns: list[re.Pattern], text: str) -> bool:
        return any(pattern.search(text) for pattern in patterns)
    def _influence(self, rule: dict, clause: str, strong_match: bool) -> Influence:
        if not strong_match:
            return Influence.UNKNOWN
        default = Influence(rule["default_influence"])
        if default is Influence.PULL:
            return Influence.PULL
        if self._matches(self.pull_patterns, clause):
            return Influence.UNKNOWN
        return default

    def classify(self, text: str, reason_context: bool = False) -> ClassificationResult:
        if not reason_context:
            return ClassificationResult()
        if reason_context:
            label = normalize_label(text)
            for item in self.broad_factors:
                if label in {normalize_label(alias) for alias in item["labels"]}:
                    return ClassificationResult([ReasonEvidence(
                        item["category"], item["subcategory"],
                        self.taxonomy.categories[item["category"]]["label"],
                    )])
        normalized = normalize(text)
        clauses = [clause.strip(" ,:") for clause in self.boundaries.split(normalized) if clause.strip(" ,:")]
        positive: dict[tuple[str, str | None], ReasonEvidence] = {}
        negative: dict[tuple[str, str | None], ReasonEvidence] = {}
        for clause in clauses:
            for rule in self.rules:
                key = rule["category"], rule["subcategory"]
                if self._matches(rule["topics"], clause) and self._matches(self.exclusions, clause):
                    negative[key] = ReasonEvidence(*key, rule["label"], supported=False)
                    continue
                if self._matches(self.uncertainty, clause):
                    continue
                match = next((found for pattern in rule["patterns"]
                              if (found := pattern.search(clause))), None)
                explicit = reason_context and normalize_label(clause) in rule["explicit_labels"]
                if match is None and not explicit:
                    continue
                # Negation inside the matched phrase can itself be evidence (e.g. "wasn't competitive").
                residual = clause if match is None else clause[:match.start()] + clause[match.end():]
                if self._matches(self.negations, residual):
                    continue
                positive[key] = ReasonEvidence(
                    *key, rule["label"], influence=self._influence(rule, clause, match is not None),
                )
        # Explicit exclusion applies to this factor across the entire response.
        return ClassificationResult(
            [item for key, item in positive.items() if key not in negative] + list(negative.values())
        )
    def clarification_for(self, text: str, reason_context: bool = False) -> dict | None:
        if reason_context:
            for clarification in self.clarifications:
                if self._matches(self._compile(clarification["patterns"]), normalize_label(text)):
                    return clarification
        return None

    def resolve_clarification(self, clarification: dict, text: str) -> ClassificationResult:
        label = normalize_label(text)
        for interpretation in clarification["interpretations"]:
            if label in {normalize_label(answer) for answer in interpretation["answers"]}:
                return ClassificationResult([ReasonEvidence(
                    interpretation["category"], interpretation["subcategory"], interpretation["label"],
                )])
        return ClassificationResult()

    def clarification_is_different(self, clarification: dict, text: str) -> bool:
        return normalize_label(text) in {
            normalize_label(answer) for answer in clarification["different_answers"]
        }

    def influence_answer(self, text: str) -> Influence | None:
        return self.influence_answers.get(normalize_label(text))
