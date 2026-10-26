import json
import re
import sys
from typing import TextIO
from conversation.state import ConversationState
from models.reasons import Influence, ReasonCandidate
from models.session import Response, ResponseDisposition, Session, SeparationContext
from nlp.intents import Intent, IntentProcessor
from nlp.processor import ClassificationResult, Processor
from nlp.taxonomy import load_knowledge
class ConversationStopped(Exception):
    pass
class ConversationEngine:
    def __init__(
        self,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        self.input_stream = input_stream if input_stream is not None else sys.stdin
        self.output_stream = output_stream if output_stream is not None else sys.stdout
        self.session = Session()
        self.state = ConversationState.WELCOME
        self.questions = load_knowledge("questions.json")
        self.intents = IntentProcessor()
        self.processor = Processor()
    def say(self, text: str) -> None:
        print(text, file=self.output_stream)
    def _reporting_guidance(self, text: str) -> bool:
        sensitive = self.intents.needs_reporting_guidance(text)
        if sensitive and self.session.reporting_state is None:
            previous_state = self.state
            self.state = ConversationState.REPORTING_GUIDANCE_REQUIRED
            self.session.reporting_state = self.state.value
            self.say(self.questions["reporting_notice"])
            self.state = previous_state
        return sensitive
    def ask(self, question_id: str, prompt: str | None = None) -> Response:
        while True:
            self.say(prompt if prompt is not None else self.questions[question_id])
            print("> ", end="", file=self.output_stream, flush=True)
            try:
                line = self.input_stream.readline()
            except KeyboardInterrupt:
                raise ConversationStopped from None
            if line == "":
                raise ConversationStopped
            text = line.rstrip("\r\n")
            intent = self.intents.detect(text, question_id)
            if intent is Intent.HELP:
                self.say(self.questions["help"])
                continue
            if intent in (Intent.STOP, Intent.EXIT) or text.strip() == "0":
                raise ConversationStopped
            disposition = ResponseDisposition.ANSWERED
            if intent is Intent.SKIP or not text.strip():
                disposition = ResponseDisposition.SKIPPED
            elif intent is Intent.PREFER_NOT_TO_ANSWER:
                disposition = ResponseDisposition.PREFER_NOT_TO_ANSWER
            response = self.session.record_response(question_id, text, disposition)
            if disposition is ResponseDisposition.ANSWERED:
                self._reporting_guidance(text)
            return response
    def _wait_for_start(self) -> None:
        self.state = ConversationState.WAITING_FOR_START
        while True:
            response = self.ask("start")
            if self.intents.detect(response.text) is Intent.START or response.text.strip() == "1":
                return
            self.say("We will begin only when you type START or enter 1.")
    def _separation_context(self) -> None:
        self.state = ConversationState.SEPARATION_CONTEXT
        while True:
            response = self.ask("context")
            if response.disposition is ResponseDisposition.PREFER_NOT_TO_ANSWER:
                self.session.separation_context = SeparationContext.PREFER_NOT_TO_ANSWER
                return
            if response.disposition is not ResponseDisposition.ANSWERED:
                return
            context = self.intents.separation_context(response.text)
            if context is not None:
                self.session.separation_context = context
                self.say(self.questions["acknowledgement"])
                return
            self.say("You may use one of the descriptions or numbers, or skip this question.")
    def _classify_response(
        self,
        response: Response,
        reason_context: bool = False,
        related_response_ids: list[str] | None = None,
        clarification_rule: dict | None = None,
    ) -> ClassificationResult:
        result = (self.processor.resolve_clarification(clarification_rule, response.text)
                  if clarification_rule is not None
                  else self.processor.classify(response.text, reason_context))
        self.session.record_evidence(response, result.evidence, related_response_ids)
        return result
    def _reason_question(self, question_id: str) -> None:
        self.state = ConversationState.EXIT_REASONS
        prompt = (self.questions["main_reason_by_context"][self.session.separation_context.value]
                  if question_id == "main_reason" and self.session.separation_context is not None
                  else None)
        response = self.ask(question_id, prompt)
        if response.disposition is not ResponseDisposition.ANSWERED:
            return
        if self.intents.has_no_more_factors(response.text, question_id):
            self.say(self.questions["acknowledgement"])
            return
        sources = [response.response_id]
        result = self._classify_response(response, reason_context=True)
        if any(item.category == "PERSONAL_CIRCUMSTANCES" for item in result.supported):
            self.say(self.questions["personal_notice"])
        if result.supported or result.excluded:
            self.say(self.questions["acknowledgement"])
            return
        if self.intents.needs_reporting_guidance(response.text):
            self.session.mark_unclassified(sources)
            return
        self.session.clarification_attempts[question_id] = 0
        clarification_rule = self.processor.clarification_for(response.text, reason_context=True)
        clarification_ids = (clarification_rule["question_ids"] if clarification_rule is not None
                             else ("clarify_first", "clarify_second"))
        different_meaning = False
        for clarification_id in clarification_ids:
            self.state = ConversationState.CLARIFYING
            prompt = (self.questions[clarification_rule["different_question_id"]]
                      if different_meaning else None)
            clarification = self.ask(clarification_id, prompt)
            self.session.clarification_attempts[question_id] += 1
            sources.append(clarification.response_id)
            if clarification.disposition is not ResponseDisposition.ANSWERED:
                break
            if self.intents.has_no_more_factors(clarification.text, question_id):
                self.say(self.questions["acknowledgement"])
                return
            if (clarification_rule is not None
                    and self.processor.clarification_is_different(clarification_rule, clarification.text)):
                different_meaning = True
                continue
            result = self._classify_response(
                clarification, True, sources[:-1],
                clarification_rule if not different_meaning else None,
            )
            if result.supported or result.excluded:
                if any(item.category == "PERSONAL_CIRCUMSTANCES" for item in result.supported):
                    self.say(self.questions["personal_notice"])
                self.say(self.questions["acknowledgement"])
                return
            if self.intents.needs_reporting_guidance(clarification.text):
                break
        self.session.mark_unclassified(sources)
        self.say(self.questions["unclassified_notice"])
    def _feedback(self) -> None:
        self.state = ConversationState.FEEDBACK
        for question_id in (
            "worked_well", "wish_different", "recommend", "return", "return_reason",
            "one_change", "not_covered",
        ):
            response = self.ask(question_id)
            if response.disposition is ResponseDisposition.ANSWERED:
                self._classify_response(response)
            self.say(self.questions["acknowledgement"])
    def _show_candidates(self, candidates: list[ReasonCandidate]) -> None:
        for number, candidate in enumerate(candidates, start=1):
            suffix = " (statements differ; both retained)" if candidate.excluding_response_ids else ""
            self.say(f"{number}. {candidate.label}{suffix}")
    def _choose_numbers(
        self, question_id: str, candidates: list[ReasonCandidate], count: int,
    ) -> tuple[list[ReasonCandidate], Response] | None:
        while True:
            self._show_candidates(candidates)
            response = self.ask(question_id)
            if response.disposition is not ResponseDisposition.ANSWERED:
                return None
            text = response.text.strip()
            if re.fullmatch(r"\d+(?:\s*,\s*\d+)*", text):
                numbers = [int(value) for value in text.split(",")]
                if (len(numbers) == count and len(set(numbers)) == count
                        and all(1 <= number <= len(candidates) for number in numbers)):
                    return [candidates[number - 1] for number in numbers], response
            self.say(f"Please enter {count} different displayed numbers separated by commas, or skip.")
    def _resolve_influence(self, candidates: list[ReasonCandidate]) -> None:
        self.state = ConversationState.RESOLVING_INFLUENCE
        for candidate in candidates:
            if (candidate.influence is not Influence.UNKNOWN
                    or candidate.category == "UNCLASSIFIED"
                    or candidate.subcategory == "PERSONAL_OTHER"):
                continue
            response = self.ask(
                "influence", self.questions["influence"].format(factor=candidate.label),
            )
            if response.disposition is ResponseDisposition.ANSWERED:
                influence = self.processor.influence_answer(response.text)
                if influence is not None:
                    candidate.add_influence(influence, response.response_id)
                else:
                    self.say("We can leave that influence unknown.")
    def _resolve_reasons(self) -> None:
        candidates = list(self.session.candidates)
        if not candidates:
            self.session.ranking_status = "NOT_APPLICABLE"
            return
        if len(candidates) == 1:
            self.session.ranking_status = "NOT_APPLICABLE"
            self._resolve_influence(candidates)
            return
        ranking_sources = []
        if len(candidates) > 3:
            self.state = ConversationState.SELECTING_REASONS
            selection = self._choose_numbers("select", candidates, 3)
            if selection is None:
                self.session.ranking_status = "SKIPPED"
                self.say("Your factors remain unranked; no primary reason will be assigned.")
                return
            candidates, response = selection
            ranking_sources.append(response.response_id)
        self.state = ConversationState.RANKING_REASONS
        ranking = self._choose_numbers("rank", candidates, len(candidates))
        if ranking is None:
            self.session.ranking_status = "SKIPPED"
            self.say("Your factors remain unranked; no primary reason will be assigned.")
            return
        candidates, response = ranking
        ranking_sources.append(response.response_id)
        self.session.rank_reasons([candidate.candidate_id for candidate in candidates], ranking_sources)
        self._resolve_influence(candidates)
    def run(self) -> Session:
        self.say(self.questions["welcome"])
        self.say(self.questions["privacy"])
        try:
            self._wait_for_start()
            self._separation_context()
            self._reason_question("main_reason")
            self._reason_question("other_factors")
            self._feedback()
            self._resolve_reasons()
            self.state = ConversationState.COMPLETED
            self.session.status = self.state.value
        except ConversationStopped:
            self.state = ConversationState.STOPPED
            self.session.status = self.state.value
            self.say(self.questions["stopped"])
        self.say(self.questions["thank_you"])
        self.say(self.questions["development_summary"])
        self.say(json.dumps(self.session.to_dict(), ensure_ascii=True, indent=2))
        return self.session
