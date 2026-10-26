import re
import unicodedata
from enum import Enum
from nltk.tokenize import RegexpTokenizer
from models.session import SeparationContext
from nlp.taxonomy import load_knowledge
_TOKENIZER = RegexpTokenizer(r"\w+(?:'\w+)*|[^\w\s]")
def normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(_TOKENIZER.tokenize(normalized.casefold()))
def normalize_label(text: str) -> str:
    return normalize(text).strip(" .!?;:,")
class Intent(str, Enum):
    START = "START"
    SKIP = "SKIP"
    PREFER_NOT_TO_ANSWER = "PREFER_NOT_TO_ANSWER"
    HELP = "HELP"
    STOP = "STOP"
    EXIT = "EXIT"
    NO_MORE_FACTORS = "NO_MORE_FACTORS"
    ANSWER = "ANSWER"
class IntentProcessor:
    def __init__(self) -> None:
        data = load_knowledge("intents.json")
        self.commands = {
            normalize_label(alias): Intent(intent)
            for intent, aliases in data["commands"].items() for alias in aliases
        }
        self.contexts = {
            normalize_label(alias): SeparationContext(context)
            for context, aliases in data["separation_contexts"].items()
            for alias in [context, *aliases]
        }
        self.no_more_factors = {normalize_label(alias) for alias in data["no_more_factors"]}
        self.sensitive_patterns = [re.compile(pattern) for pattern in data["sensitive_topics"]]
    def detect(self, text: str, question_id: str | None = None) -> Intent:
        if self.has_no_more_factors(text, question_id):
            return Intent.NO_MORE_FACTORS
        return self.commands.get(normalize_label(text), Intent.ANSWER)

    def separation_context(self, text: str) -> SeparationContext | None:
        return self.contexts.get(normalize_label(text))

    def has_no_more_factors(self, text: str, question_id: str | None = None) -> bool:
        return question_id == "other_factors" and normalize_label(text) in self.no_more_factors

    def needs_reporting_guidance(self, text: str) -> bool:
        return any(pattern.search(normalize(text)) for pattern in self.sensitive_patterns)
