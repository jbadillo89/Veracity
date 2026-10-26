import ast
import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from uuid import UUID

import nltk

from Veracity_M1 import main
from conversation.engine import ConversationEngine
from conversation.state import ConversationState
from models.reasons import ClassificationStatus, Influence, ReasonRole
from models.session import Response, ResponseDisposition, Session, SeparationContext
from nlp.taxonomy import load_knowledge


PROJECT = Path(__file__).resolve().parent.parent
WORKSPACE = PROJECT.parent.parent
FEEDBACK = ["The tools were useful.", "Clearer processes.", "Perhaps.", "No.",
            "A different direction.", "Improve processes.", "Nothing else."]


def converse(lines: list[str]) -> tuple[Session, str, ConversationEngine]:
    output = io.StringIO()
    engine = ConversationEngine(io.StringIO("\n".join(lines) + "\n"), output)
    session = engine.run()
    return session, output.getvalue(), engine


def standard_lines(reason: str = "The salary was too low.", other: str = "No.") -> list[str]:
    return ["START", "voluntary", reason, other, *FEEDBACK]


class PrototypeTests(unittest.TestCase):
    def test_application_starts(self):
        result = subprocess.run(
            [sys.executable, "-B", str(PROJECT / "Veracity_M1.py")],
            input="STOP\n", text=True, capture_output=True, cwd=WORKSPACE, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Welcome to Veracity", result.stdout)
        self.assertIn('"status": "STOPPED"', result.stdout)

    def test_nltk_imports(self):
        self.assertTrue(nltk.__version__)
        self.assertEqual(nltk.tokenize.RegexpTokenizer(r"\w+").tokenize("plain text"),
                         ["plain", "text"])

    def test_uses_requested_local_python_312(self):
        self.assertEqual(sys.version_info[:2], (3, 12))
        self.assertEqual(Path(sys.executable).resolve(),
                         (WORKSPACE / ".venv" / "Scripts" / "python.exe").resolve())
        self.assertNotEqual(sys.prefix, sys.base_prefix)

    def test_full_conversation_completes(self):
        session, output, engine = converse(standard_lines(other="The hours were terrible.") + ["1,2"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(engine.state, ConversationState.COMPLETED)
        self.assertEqual(session.separation_context, SeparationContext.VOLUNTARY)
        self.assertEqual(len(session.reasons), 2)
        self.assertEqual(session.reasons[0].role, ReasonRole.PRIMARY)
        self.assertIn("Development/testing only", output)
        self.assertEqual([r.question_id for r in session.responses], [
            "start", "context", "main_reason", "other_factors", "worked_well",
            "wish_different", "recommend", "return", "return_reason", "one_change",
            "not_covered", "rank",
        ])

    def test_main_entrypoint_uses_real_engine(self):
        output = io.StringIO()
        session = main(io.StringIO("EXIT\n"), output)
        self.assertEqual(session.status, "STOPPED")
        self.assertIn("No more questions", output.getvalue())

    def test_random_sessions_are_independent(self):
        first, second = Session(), Session()
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual(UUID(first.session_id).version, 4)
        self.assertEqual(UUID(second.session_id).version, 4)
        first.record_response("main_reason", "personal reasons")
        self.assertEqual(second.responses, [])
        self.assertIsNot(first.candidates, second.candidates)
        self.assertIsNot(first.clarification_attempts, second.clarification_attempts)

    def test_no_identity_fields_are_collected(self):
        prohibited = {"name", "participant_name", "employee_number", "email", "email_address",
                      "windows_username", "username", "machine_identifier", "hostname",
                      "ip_address", "manager_name", "coworker_name", "new_employer_name"}
        for model in (Session, Response):
            self.assertTrue(prohibited.isdisjoint({field.name for field in fields(model)}))
        session, output, _ = converse(["STOP"])
        self.assertTrue(prohibited.isdisjoint(session.to_dict()))
        self.assertIn("Please do not provide names", output)

    def test_skip_works(self):
        session, _, _ = converse(standard_lines("SKIP")[:-1] + ["SKIP"])
        self.assertEqual(session.status, "COMPLETED")
        response = next(r for r in session.responses if r.question_id == "main_reason")
        self.assertEqual(response.disposition, ResponseDisposition.SKIPPED)
        self.assertEqual(session.reasons, [])
        self.assertEqual(session.clarification_attempts, {})

    def test_prefer_not_to_answer_works(self):
        session, _, _ = converse(["START", "PREFER NOT TO ANSWER", "prefer not to answer",
                                  "prefer not to answer", *( ["SKIP"] * 7 )])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.separation_context, SeparationContext.PREFER_NOT_TO_ANSWER)
        self.assertEqual(session.responses[2].disposition,
                         ResponseDisposition.PREFER_NOT_TO_ANSWER)
        self.assertEqual(session.candidates, [])

    def test_help_works_without_advancing(self):
        lines = standard_lines()
        lines.insert(2, "HELP")
        session, output, _ = converse(lines)
        self.assertEqual(session.status, "COMPLETED")
        self.assertIn("You control what to share", output)
        self.assertNotIn("HELP", [r.text for r in session.responses])
        self.assertEqual(sum(r.question_id == "main_reason" for r in session.responses), 1)

    def test_stop_works_immediately(self):
        session, output, _ = converse(["START", "voluntary", "STOP", "must not be read"])
        self.assertEqual(session.status, "STOPPED")
        self.assertNotIn("must not be read", [r.text for r in session.responses])
        self.assertNotIn(load_knowledge("questions.json")["other_factors"], output)

    def test_exit_works_immediately(self):
        session, _, engine = converse(["EXIT", "START"])
        self.assertEqual(session.status, "STOPPED")
        self.assertEqual(engine.state, ConversationState.STOPPED)
        self.assertEqual(session.responses, [])

    def test_eof_stops_without_traceback(self):
        engine = ConversationEngine(io.StringIO(""), io.StringIO())
        self.assertEqual(engine.run().status, "STOPPED")

    def test_waits_for_explicit_start(self):
        session, output, _ = converse(["hello", "SKIP", "HELP", "STOP"])
        self.assertEqual(session.status, "STOPPED")
        self.assertNotIn("context", [r.question_id for r in session.responses])
        self.assertIn("begin only when you type START", output)

    def test_blank_answer_is_skipped(self):
        session, _, _ = converse(standard_lines("") )
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.responses[2].disposition, ResponseDisposition.SKIPPED)

    def test_separation_context_is_independent_for_all_values(self):
        for context in SeparationContext:
            with self.subTest(context=context):
                session, _, _ = converse(["START", context.value, "SKIP", "SKIP",
                                          *( ["SKIP"] * 7 )])
                self.assertEqual(session.separation_context, context)
                self.assertEqual(session.candidates, [])
                self.assertEqual(session.reasons, [])

    def test_invalid_context_does_not_create_exit_reason(self):
        session, _, _ = converse(["START", "The salary was too low.", "retirement",
                                  "SKIP", "SKIP", *( ["SKIP"] * 7 )])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.separation_context, SeparationContext.RETIREMENT)
        self.assertEqual(session.candidates, [])

    def test_ambiguous_response_stays_unclassified_after_two_questions(self):
        session, output, _ = converse([
            "START", "voluntary", "It was complicated.", "Hard to describe.",
            "Still unclear.", "No.", *FEEDBACK,
        ])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.clarification_attempts, {"main_reason": 2})
        self.assertEqual(session.candidates[0].category, "UNCLASSIFIED")
        self.assertEqual(session.candidates[0].status, ClassificationStatus.UNCLASSIFIED)
        self.assertEqual(session.candidates[0].influence, Influence.UNKNOWN)
        self.assertEqual(session.reasons, [])
        self.assertEqual(session.ranking_status, "NOT_APPLICABLE")
        self.assertIn("unspecified", output)

    def test_help_does_not_consume_clarification_attempts(self):
        session, _, _ = converse([
            "START", "voluntary", "Complicated.", "HELP", "Still unclear.", "HELP",
            "Unclear.", "No.", *FEEDBACK,
        ])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.clarification_attempts["main_reason"], 2)
        self.assertEqual(sum(r.question_id.startswith("clarify") for r in session.responses), 2)

    def test_clarification_can_resolve_with_human_label(self):
        session, output, _ = converse([
            "START", "voluntary", "Things changed.", "Not sure how to describe it.",
            "career growth and development", "No.", *FEEDBACK, "push",
        ])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.candidates[0].category, "CAREER_DEVELOPMENT")
        self.assertEqual(session.candidates[0].influence, Influence.PUSH)
        self.assertIn("R3", session.candidates[0].source_response_ids)
        self.assertIn("R5", session.candidates[0].source_response_ids)
        self.assertEqual(session.reasons, [])
        prompts = output.split("Development/testing only")[0]
        self.assertNotIn("CAREER_DEVELOPMENT", prompts)

    def test_skip_clarification_keeps_original_ambiguous_response(self):
        session, _, _ = converse([
            "START", "voluntary", "Complicated.", "SKIP", "No.", *FEEDBACK,
        ])
        self.assertEqual(session.clarification_attempts["main_reason"], 1)
        self.assertEqual(session.responses[2].text, "Complicated.")
        self.assertEqual(session.candidates[0].status, ClassificationStatus.UNCLASSIFIED)
        self.assertEqual(session.reasons, [])

    def test_both_reason_questions_have_maximum_two_clarifications(self):
        session, _, _ = converse([
            "START", "voluntary", "Unclear.", "Unclear.", "Unclear.",
            "Unclear.", "Unclear.", "Unclear.", *FEEDBACK, "SKIP",
        ])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.clarification_attempts, {"main_reason": 2, "other_factors": 2})

    def test_more_than_three_requires_selection_then_ranking(self):
        reason = ("The salary was too low. There was nowhere for me to advance. "
                  "The hours were terrible. I had problems with my direct supervisor. "
                  "I'm relocating.")
        session, output, _ = converse(standard_lines(reason) + ["5,3,2", "2,3,1", "unknown"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(len(session.candidates), 5)
        self.assertEqual(len(session.reasons), 3)
        self.assertEqual([r.subcategory for r in session.reasons],
                         ["WORKING_HOURS", "CAREER_PROGRESSION", "RELOCATION"])
        self.assertEqual([r.rank for r in session.reasons], [1, 2, 3])
        self.assertEqual([r.role for r in session.reasons],
                         [ReasonRole.PRIMARY, ReasonRole.CONTRIBUTING, ReasonRole.CONTRIBUTING])
        self.assertEqual(session.responses[2].text, reason)
        self.assertIn("more than three factors", output)
        self.assertLess([r.question_id for r in session.responses].index("select"),
                        [r.question_id for r in session.responses].index("rank"))

    def test_participant_controls_ranking_not_mention_order(self):
        session, _, _ = converse(standard_lines(
            "The salary was too low and the hours were terrible.") + ["2,1"])
        self.assertEqual(session.reasons[0].subcategory, "WORKING_HOURS")
        self.assertEqual(session.reasons[1].subcategory, "SALARY")

    def test_skip_selection_does_not_assign_primary(self):
        reason = ("The salary was too low. The hours were terrible. "
                  "There was nowhere to advance. I'm relocating.")
        session, output, _ = converse(standard_lines(reason) + ["SKIP"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(len(session.candidates), 4)
        self.assertEqual(session.reasons, [])
        self.assertEqual(session.ranking_status, "SKIPPED")
        self.assertIn("no primary reason", output)

    def test_skip_ranking_does_not_assign_primary(self):
        session, _, _ = converse(standard_lines(other="The hours were terrible.")
                                  + ["PREFER NOT TO ANSWER"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.reasons, [])
        self.assertEqual(len(session.candidates), 2)
        self.assertEqual(session.ranking_status, "SKIPPED")

    def test_invalid_rank_is_not_accepted(self):
        session, output, _ = converse(standard_lines(
            "The salary was too low and the hours were terrible.") + ["1,1", "9,1", "2,1"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.reasons[0].subcategory, "WORKING_HOURS")
        self.assertIn("2 different displayed numbers", output)

    def test_invalid_selection_cannot_bypass_three_factor_limit(self):
        reason = ("The salary was too low. The hours were terrible. "
                  "There was nowhere to advance. I'm relocating.")
        session, _, _ = converse(standard_lines(reason) + ["1,2,3,4", "1,2,3", "3,2,1"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(len(session.reasons), 3)
        self.assertEqual(session.reasons[0].subcategory, "CAREER_PROGRESSION")

    def test_personal_reasons_do_not_require_details(self):
        session, output, _ = converse(standard_lines("personal reasons"))
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.candidates[0].subcategory, "PERSONAL_OTHER")
        self.assertEqual(session.candidates[0].influence, Influence.UNKNOWN)
        self.assertEqual(session.reasons, [])
        self.assertEqual(session.clarification_attempts, {})
        self.assertNotIn("influence", [r.question_id for r in session.responses])
        self.assertIn("no further explanation is needed", output)

    def test_sensitive_disclosure_preserves_feedback_without_investigation(self):
        disclosure = "I experienced harassment. I do not want to provide details."
        session, output, _ = converse(standard_lines(disclosure))
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.responses[2].text, disclosure)
        self.assertEqual(session.reporting_state, "REPORTING_GUIDANCE_REQUIRED")
        self.assertEqual(session.clarification_attempts, {})
        prompts = output.split("Development/testing only")[0]
        self.assertIn("appropriate reporting channels", prompts)
        self.assertNotIn("http", prompts)
        self.assertNotIn("@", prompts)
        self.assertNotIn("Who", prompts)
        self.assertNotIn("accused", prompts)

    def test_sensitive_disclosure_in_later_feedback_gets_notice_once(self):
        feedback = ["I experienced bullying.", "There was harassment.", *FEEDBACK[2:]]
        session, output, _ = converse(["START", "voluntary", "SKIP", "No.", *feedback])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.reporting_state, "REPORTING_GUIDANCE_REQUIRED")
        self.assertEqual(output.count("Where applicable, please use appropriate reporting channels"), 1)
        self.assertIn("I experienced bullying.", [r.text for r in session.responses])

    def test_unknown_influence_asks_neutrally(self):
        session, output, _ = converse(standard_lines("salary") + ["pull"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.candidates[0].influence, Influence.PULL)
        self.assertEqual(session.reasons, [])
        self.assertIn("external attraction", output)
        self.assertTrue(session.candidates[0].influence_response_ids)

    def test_relocation_influence_is_optional_and_not_inferred(self):
        session, _, _ = converse(standard_lines("I'm relocating.") + ["SKIP"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertIn("influence", [r.question_id for r in session.responses])
        self.assertEqual(session.candidates[0].influence, Influence.UNKNOWN)

    def test_unclear_influence_is_not_guessed(self):
        session, _, _ = converse(standard_lines("salary") + ["it is complicated"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.candidates[0].influence, Influence.UNKNOWN)

    def test_skip_influence_is_allowed(self):
        session, _, _ = converse(standard_lines("salary") + ["SKIP"])
        self.assertEqual(session.candidates[0].influence, Influence.UNKNOWN)
        self.assertEqual(session.status, "COMPLETED")

    def test_explicit_both_supports_mixed_influence(self):
        session, _, _ = converse(standard_lines("salary") + ["both"])
        self.assertEqual(session.candidates[0].influence, Influence.MIXED)

    def test_stop_is_available_during_clarification_and_ranking(self):
        for lines in (["START", "voluntary", "Unclear.", "STOP"],
                      standard_lines(other="The hours were terrible.") + ["EXIT"]):
            with self.subTest(lines=lines):
                session, _, _ = converse(lines)
                self.assertEqual(session.status, "STOPPED")
                self.assertEqual(session.reasons, [])

    def test_contradictory_feedback_keeps_both_statements(self):
        earlier = "Salary was one of my reasons."
        later = "Actually salary wasn't really why I left."
        session, _, _ = converse(standard_lines(earlier, later))
        self.assertEqual(session.candidates[0].status, ClassificationStatus.CONFLICTING)
        self.assertEqual(session.reasons, [])
        self.assertEqual(session.responses[2].text, earlier)
        self.assertEqual(session.responses[3].text, later)
        self.assertIn("R3", session.candidates[0].source_response_ids)
        self.assertIn("R4", session.candidates[0].source_response_ids)

    def test_console_summary_is_valid_json(self):
        session, output, _ = converse(standard_lines(other="The hours were terrible.") + ["1,2"])
        marker = load_knowledge("questions.json")["development_summary"] + "\n"
        summary = json.loads(output.split(marker, 1)[1])
        self.assertEqual(summary["session_id"], session.session_id)
        self.assertEqual(summary["reasons"][0]["rank"], 1)

    def test_no_participant_data_is_written(self):
        before = {p.relative_to(PROJECT): p.read_bytes() for p in PROJECT.rglob("*") if p.is_file()}
        with tempfile.TemporaryDirectory(prefix="veracity_test_", dir=WORKSPACE) as directory:
            result = subprocess.run(
                [sys.executable, "-B", str(PROJECT / "Veracity_M1.py")],
                input="\n".join(standard_lines()) + "\n", text=True,
                capture_output=True, cwd=directory, timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"status": "COMPLETED"', result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])
        after = {p.relative_to(PROJECT): p.read_bytes() for p in PROJECT.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_application_has_no_prohibited_service_imports(self):
        forbidden = {"flask", "django", "fastapi", "sqlite3", "sqlalchemy", "openai",
                     "qdrant_client", "tkinter", "requests", "socket", "logging"}
        for source in PROJECT.rglob("*.py"):
            if "tests" in source.parts:
                continue
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                         else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                self.assertTrue(forbidden.isdisjoint({name.split(".")[0] for name in names}), source)


class ConversationCorrectionTests(unittest.TestCase):
    def assert_no_more_factors(self, answer):
        session, output, _ = converse(standard_lines(other=answer))
        self.assertEqual(session.status, "COMPLETED")
        other = next(r for r in session.responses if r.question_id == "other_factors")
        self.assertEqual(other.text, answer)
        self.assertEqual(other.evidence, [])
        self.assertNotIn("other_factors", session.clarification_attempts)
        self.assertEqual(len(session.candidates), 1)
        self.assertEqual(session.candidates[0].subcategory, "SALARY")
        self.assertNotIn(other.response_id, session.candidates[0].source_response_ids)
        question_ids = [r.question_id for r in session.responses]
        self.assertEqual(question_ids[question_ids.index("other_factors") + 1], "worked_well")
        after_question = output.split(load_knowledge("questions.json")["other_factors"], 1)[1]
        self.assertTrue(after_question.startswith("\n> Understood. Thank you.\nWhat worked well"))

    def test_not_really_ends_additional_factor_question(self):
        self.assert_no_more_factors("not really")

    def test_no_ends_additional_factor_question(self):
        self.assert_no_more_factors("no")

    def test_none_ends_additional_factor_question(self):
        self.assert_no_more_factors("none")

    def test_nothing_else_ends_additional_factor_question(self):
        self.assert_no_more_factors("nothing else")

    def test_other_requested_no_more_factor_aliases(self):
        for answer in ["nope", "no more", "no other reason", "no other reasons",
                       "that's all", "that is all", "  NOT REALLY!  ", "that’s all"]:
            with self.subTest(answer=answer):
                self.assert_no_more_factors(answer)

    def test_no_more_factors_creates_no_candidate(self):
        session, _, _ = converse(["START", "voluntary", "SKIP", "not really", *(["SKIP"] * 7)])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.candidates, [])
        self.assertEqual(session.reasons, [])
        self.assertEqual(session.responses[3].text, "not really")

    def test_no_to_main_reason_is_not_no_more_factors(self):
        session, _, _ = converse([
            "START", "voluntary", "no", "Hard to explain.", "Still unclear.",
            "not really", *FEEDBACK,
        ])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.clarification_attempts, {"main_reason": 2})
        self.assertEqual(session.candidates[0].category, "UNCLASSIFIED")
        self.assertEqual(session.responses[2].text, "no")

    def test_no_during_additional_factor_clarification_also_ends_it(self):
        session, _, _ = converse([
            "START", "voluntary", "salary", "Hard to describe another factor.", "no",
            *FEEDBACK, "unknown",
        ])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.clarification_attempts, {"other_factors": 1})
        self.assertEqual(len(session.candidates), 1)
        self.assertEqual(session.candidates[0].subcategory, "SALARY")
        self.assertEqual(session.responses[4].text, "no")
        self.assertEqual(session.responses[4].evidence, [])

    def test_new_career_opportunities_triggers_targeted_clarification(self):
        session, output, _ = converse(["START", "voluntary", "new career opportunities", "STOP"])
        self.assertEqual(session.status, "STOPPED")
        self.assertEqual(session.candidates, [])
        self.assertIn("When you mention career opportunities", output)
        self.assertIn("missing here", output)
        self.assertIn("somewhere else", output)
        self.assertIn("something different", output)
        self.assertNotIn(load_knowledge("questions.json")["clarify_first"], output)
        prompts = output.split(load_knowledge("questions.json")["development_summary"])[0]
        for identifier in ["CAREER_DEVELOPMENT", "EXTERNAL_OPPORTUNITY", "CAREER_ADVANCEMENT"]:
            self.assertNotIn(identifier, prompts)
        self.assertEqual(session.responses[2].text, "new career opportunities")

    def test_career_clarification_accepts_internal_or_external_without_guessing_influence(self):
        for answer, category, subcategory in [
            ("missing here", "CAREER_DEVELOPMENT", "CAREER_PROGRESSION"),
            ("somewhere else", "EXTERNAL_OPPORTUNITY", "CAREER_ADVANCEMENT"),
        ]:
            with self.subTest(answer=answer):
                session, _, _ = converse([
                    "START", "voluntary", "new career opportunities", answer, "not really",
                    *FEEDBACK, "unknown",
                ])
                self.assertEqual(session.status, "COMPLETED")
                self.assertEqual(len(session.candidates), 1)
                self.assertEqual((session.candidates[0].category, session.candidates[0].subcategory),
                                 (category, subcategory))
                self.assertEqual(session.candidates[0].influence, Influence.UNKNOWN)
                self.assertEqual(session.reasons, [])
                self.assertEqual(session.clarification_attempts, {"main_reason": 1})
                self.assertIn("R3", session.candidates[0].source_response_ids)
                self.assertIn("R4", session.candidates[0].source_response_ids)
                self.assertEqual(session.responses[2].text, "new career opportunities")
                self.assertEqual(session.responses[3].text, answer)

    def test_career_clarification_does_not_silently_drift_to_salary(self):
        session, output, _ = converse([
            "START", "voluntary", "new career opportunities", "salary", "still unclear",
            "not really", *FEEDBACK,
        ])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.candidates[0].category, "UNCLASSIFIED")
        self.assertEqual(len(session.candidates), 1)
        self.assertEqual(session.clarification_attempts, {"main_reason": 2})
        self.assertIn("Still referring to the career opportunities", output)
        self.assertEqual(session.responses[3].text, "salary")

    def test_something_different_can_voluntarily_change_the_factor(self):
        session, _, _ = converse([
            "START", "voluntary", "new career opportunities", "something different", "salary",
            "not really", *FEEDBACK, "unclear",
        ])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.candidates[0].subcategory, "SALARY")
        self.assertEqual(session.candidates[0].influence, Influence.UNKNOWN)
        self.assertEqual(session.reasons, [])
        self.assertEqual(session.clarification_attempts, {"main_reason": 2})

    def test_targeted_clarification_allows_skip_and_prefer_not_to_answer(self):
        for command in ["SKIP", "PREFER NOT TO ANSWER"]:
            with self.subTest(command=command):
                session, _, _ = converse([
                    "START", "voluntary", "new career opportunities", command,
                    "not really", *FEEDBACK,
                ])
                self.assertEqual(session.status, "COMPLETED")
                self.assertEqual(session.candidates[0].status, ClassificationStatus.UNCLASSIFIED)
                self.assertEqual(session.reasons, [])
                self.assertEqual(session.responses[3].text, command)

    def test_targeted_clarification_allows_help_stop_and_exit(self):
        for command in ["STOP", "EXIT"]:
            with self.subTest(command=command):
                session, output, _ = converse([
                    "START", "voluntary", "new career opportunities", "HELP", command,
                ])
                self.assertEqual(session.status, "STOPPED")
                self.assertEqual(session.candidates, [])
                self.assertIn("You control what to share", output)

    def test_feedback_topic_mentions_remain_feedback_only(self):
        feedback = ["my boss", "my salary", "no", "no", "unclear",
                    "the compensation package", "nothing else"]
        session, _, _ = converse(["START", "voluntary", "SKIP", "not really", *feedback])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.candidates, [])
        self.assertEqual(session.reasons, [])
        for question_id, text in zip(["worked_well", "wish_different", "recommend", "return",
                                      "return_reason", "one_change", "not_covered"], feedback):
            response = next(r for r in session.responses if r.question_id == question_id)
            self.assertEqual(response.text, text)
            self.assertEqual(response.evidence, [])

    def test_salary_external_ranking_and_unknown_influences_remain_participant_controlled(self):
        session, _, _ = converse(standard_lines("salary", "external opportunity")
                                  + ["1,2", "unclear", "unknown"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual([r.category for r in session.reasons],
                         ["COMPENSATION_BENEFITS", "EXTERNAL_OPPORTUNITY"])
        self.assertEqual([r.rank for r in session.reasons], [1, 2])
        self.assertEqual([r.role for r in session.reasons], [ReasonRole.PRIMARY, ReasonRole.CONTRIBUTING])
        self.assertEqual([r.influence for r in session.reasons], [Influence.UNKNOWN, Influence.UNKNOWN])
        reversed_session, _, _ = converse(standard_lines("salary", "external opportunity")
                                           + ["2,1", "unknown", "unclear"])
        self.assertEqual(reversed_session.reasons[0].category, "EXTERNAL_OPPORTUNITY")


class StrongFeedbackIsolationTests(unittest.TestCase):
    def test_strong_phrases_in_every_feedback_question_create_no_candidates(self):
        question_ids = ["worked_well", "wish_different", "recommend", "return",
                        "return_reason", "one_change", "not_covered"]
        phrases = ["  The salary was too low.  ", "The hours are terrible.",
                   "I had problems with my direct supervisor.",
                   "There was nowhere for me to advance.", "My manager was excellent."]
        for index, question_id in enumerate(question_ids):
            for text in phrases:
                with self.subTest(question=question_id, text=text):
                    feedback = ["SKIP"] * 7
                    feedback[index] = text
                    session, _, _ = converse(["START", "voluntary", "SKIP", "not really", *feedback])
                    self.assertEqual(session.status, "COMPLETED")
                    self.assertEqual(session.candidates, [])
                    self.assertEqual(session.reasons, [])
                    self.assertEqual(session.ranking_status, "NOT_APPLICABLE")
                    response = next(r for r in session.responses if r.question_id == question_id)
                    self.assertEqual(response.text, text)
                    self.assertEqual(response.evidence, [])

    def test_feedback_cannot_change_existing_reason_sources_or_influence(self):
        feedback = ["My manager was excellent.", "The salary was too low.",
                    "There was nowhere for me to advance.", "The hours were terrible.",
                    "I had problems with my direct supervisor.",
                    "I accepted another job because it pays more.", "I'm relocating."]
        session, _, _ = converse(["START", "voluntary", "salary", "not really",
                                 *feedback, "unknown"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(len(session.candidates), 1)
        self.assertEqual(session.candidates[0].subcategory, "SALARY")
        self.assertEqual(session.candidates[0].supporting_response_ids, ["R3"])
        self.assertEqual(session.candidates[0].source_response_ids, ["R3"])
        self.assertEqual(session.candidates[0].influence, Influence.UNKNOWN)
        self.assertEqual(session.reasons, [])
        for response, original in zip(session.responses[4:11], feedback):
            self.assertEqual(response.text, original)
            self.assertEqual(response.evidence, [])

    def test_strong_reason_answers_still_allow_participant_ranking(self):
        session, _, _ = converse(standard_lines("The salary was too low.", "The hours were terrible.")
                                  + ["2,1"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual([r.subcategory for r in session.reasons], ["WORKING_HOURS", "SALARY"])
        self.assertEqual([r.rank for r in session.reasons], [1, 2])
        self.assertEqual([r.role for r in session.reasons], [ReasonRole.PRIMARY, ReasonRole.CONTRIBUTING])
        self.assertEqual([r.influence for r in session.reasons], [Influence.PUSH, Influence.PUSH])

    def test_sensitive_feedback_is_preserved_without_creating_reason_candidates(self):
        text = "I experienced harassment. The hours were terrible."
        feedback = ["SKIP", text, *(["SKIP"] * 5)]
        session, output, _ = converse(["START", "voluntary", "SKIP", "not really", *feedback])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.reporting_state, "REPORTING_GUIDANCE_REQUIRED")
        self.assertIn("appropriate reporting channels", output)
        self.assertEqual(session.candidates, [])
        response = next(r for r in session.responses if r.question_id == "wish_different")
        self.assertEqual(response.text, text)
        self.assertEqual(response.evidence, [])


class SeparationConversationTests(unittest.TestCase):
    def test_all_separation_contexts_use_their_own_main_prompt(self):
        expected = {
            SeparationContext.VOLUNTARY: "What led you to decide to leave?",
            SeparationContext.INVOLUNTARY: "In your own words, what circumstances led to the separation?",
            SeparationContext.RETIREMENT: "What factors influenced your decision to retire at this time?",
            SeparationContext.CONTRACT_END: "Was the contract ending the main circumstance, or were there other factors you would like to share?",
            SeparationContext.INTERNAL_TRANSFER: "What factors influenced your internal move?",
            SeparationContext.OTHER: "What would you like to share about the circumstances surrounding the separation?",
            SeparationContext.PREFER_NOT_TO_ANSWER: "What would you like to share about the circumstances of leaving?",
        }
        for context, wording in expected.items():
            with self.subTest(context=context):
                session, output, _ = converse(["START", context.value, "STOP"])
                self.assertEqual(session.status, "STOPPED")
                self.assertIn("\n" + wording + "\n> ", output)
                self.assertEqual(session.candidates, [])

    def test_involuntary_wording_does_not_assume_a_decision_to_leave(self):
        _, output, _ = converse(["START", "INVOLUNTARY", "SKIP", "not really", *(["SKIP"] * 7)])
        prompts = output.split("Development/testing only")[0]
        self.assertIn("circumstances led to the separation", prompts)
        self.assertNotIn("decide to leave", prompts)
        self.assertNotIn("decided to leave", prompts)
        self.assertNotIn("decision to leave", prompts)

    def test_involuntary_context_alone_creates_no_departure_candidate(self):
        session, _, _ = converse(["START", "INVOLUNTARY", "SKIP", "not really", *(["SKIP"] * 7)])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.separation_context, SeparationContext.INVOLUNTARY)
        self.assertEqual(session.candidates, [])
        self.assertEqual(session.reasons, [])
        self.assertEqual(session.responses[1].evidence, [])

    def test_voluntary_retains_voluntary_decision_wording(self):
        _, output, _ = converse(["START", "VOLUNTARY", "STOP"])
        prompts = output.split("Development/testing only")[0]
        self.assertIn("\nWhat led you to decide to leave?\n> ", prompts)
        self.assertNotIn("circumstances led to the separation", prompts)

    def test_employer_event_evidence_is_independent_of_separation_context(self):
        for context in SeparationContext:
            with self.subTest(context=context):
                session, _, _ = converse(["START", context.value, "I was fired", "No",
                                          *(["SKIP"] * 7), "unknown"])
                self.assertEqual(session.status, "COMPLETED")
                self.assertEqual(session.separation_context, context)
                self.assertEqual(len(session.candidates), 1)
                self.assertEqual(session.candidates[0].key,
                                 ("EMPLOYMENT_EVENT", "EMPLOYER_INITIATED"))
                self.assertEqual(session.candidates[0].source_response_ids, ["R3"])
                self.assertEqual(session.responses[1].evidence, [])

    def test_fired_then_no_preserves_responses_without_clarification_or_ranking(self):
        lines = ["START", "INVOLUNTARY", "I was fired", "No", *FEEDBACK, "unknown"]
        session, output, engine = converse(lines)
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(engine.state, ConversationState.COMPLETED)
        self.assertEqual([response.text for response in session.responses], lines)
        self.assertEqual(session.clarification_attempts, {})
        self.assertEqual(len(session.candidates), 1)
        self.assertEqual(session.candidates[0].key, ("EMPLOYMENT_EVENT", "EMPLOYER_INITIATED"))
        self.assertEqual(session.candidates[0].source_response_ids, ["R3"])
        self.assertEqual(session.candidates[0].influence, Influence.UNKNOWN)
        self.assertEqual(session.ranking_status, "NOT_APPLICABLE")
        self.assertEqual(session.ranked_candidate_ids, [])
        self.assertEqual(session.ranking_response_ids, [])
        self.assertEqual(session.reasons, [])
        self.assertEqual(session.responses[3].evidence, [])
        prompts = output.split("Development/testing only")[0]
        self.assertIn("In your own words, what circumstances led to the separation?", prompts)
        self.assertNotIn("Please rank", prompts)
        self.assertNotIn(load_knowledge("questions.json")["clarify_first"], prompts)
        for inferred_cause in ["misconduct", "theft", "performance", "disciplinary",
                               "policy violation", "attendance", "guilty", "wrongdoing"]:
            self.assertNotIn(inferred_cause, prompts.casefold())

    def test_supported_and_unclassified_factors_still_allow_participant_ranking(self):
        session, output, _ = converse(["START", "INVOLUNTARY", "I was fired",
                                      "Another unclear factor", "SKIP", *(["SKIP"] * 7),
                                      "2,1", "unknown"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual([reason.category for reason in session.reasons],
                         ["UNCLASSIFIED", "EMPLOYMENT_EVENT"])
        self.assertEqual(session.ranking_status, "PARTICIPANT_RANKED")
        self.assertEqual(session.ranked_candidate_ids, ["F2", "F1"])
        self.assertIn("Please rank", output)

    def test_explicit_fired_statement_completes_without_singleton_ranking(self):
        session, output, _ = converse(["START", "INVOLUNTARY", "I was fired", "not really",
                                      *(["SKIP"] * 7), "unknown"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(len(session.candidates), 1)
        candidate = session.candidates[0]
        self.assertEqual((candidate.category, candidate.subcategory),
                         ("EMPLOYMENT_EVENT", "EMPLOYER_INITIATED"))
        self.assertEqual(candidate.influence, Influence.UNKNOWN)
        self.assertEqual(candidate.supporting_response_ids, ["R3"])
        self.assertEqual(session.responses[2].text, "I was fired")
        self.assertEqual(session.reasons, [])
        self.assertEqual(session.ranking_status, "NOT_APPLICABLE")
        self.assertNotIn("rank", [r.question_id for r in session.responses])
        self.assertNotIn("Please rank", output)
        self.assertNotIn("1. unspecified reason", output)

    def test_volunteered_misconduct_stays_unclassified_and_is_preserved(self):
        for text in ["I cursed at someone", "I robbed the company", "I was a thief", "I broke company rules"]:
            with self.subTest(text=text):
                lines = ["START", "INVOLUNTARY", text, "SKIP", "not really", *(["SKIP"] * 7)]
                session, output, _ = converse(lines)
                self.assertEqual(session.status, "COMPLETED")
                self.assertEqual([response.text for response in session.responses], lines)
                self.assertEqual(len(session.candidates), 1)
                self.assertEqual(session.candidates[0].key, ("UNCLASSIFIED", None))
                self.assertEqual(session.candidates[0].status, ClassificationStatus.UNCLASSIFIED)
                self.assertEqual(session.candidates[0].influence, Influence.UNKNOWN)
                self.assertEqual(set(session.candidates[0].source_response_ids), {"R3", "R4"})
                self.assertEqual(session.ranking_status, "NOT_APPLICABLE")
                self.assertEqual(session.ranked_candidate_ids, [])
                self.assertEqual(session.ranking_response_ids, [])
                self.assertEqual(session.reasons, [])
                prompts = output.split("Development/testing only")[0]
                self.assertNotIn("Please rank", prompts)
                self.assertNotIn("Who", prompts)
                self.assertNotIn("guilty", prompts.casefold())
                self.assertNotIn("dishonest", prompts.casefold())

    def test_two_unresolved_attempts_keep_separate_clarification_sources(self):
        session, _, _ = converse([
            "START", "INVOLUNTARY", "First circumstances are hard to describe.",
            "More about the first.", "Still unclear.",
            "Something else affected me but is difficult to describe.", "More about the second.",
            "Still unclear.", *(["SKIP"] * 7), "SKIP",
        ])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(len(session.candidates), 2)
        self.assertEqual(set(session.candidates[0].source_response_ids), {"R3", "R4", "R5"})
        self.assertEqual(set(session.candidates[1].source_response_ids), {"R6", "R7", "R8"})
        self.assertEqual(session.clarification_attempts, {"main_reason": 2, "other_factors": 2})
        self.assertEqual(session.reasons, [])

    def test_single_unclassified_candidate_closes_without_ranking(self):
        session, output, _ = converse(["START", "OTHER", "Hard to describe.", "Still unclear.",
                                      "Still unclear.", "not really", *(["SKIP"] * 7)])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(len(session.candidates), 1)
        self.assertEqual(session.candidates[0].category, "UNCLASSIFIED")
        self.assertEqual(set(session.candidates[0].source_response_ids), {"R3", "R4", "R5"})
        self.assertEqual(session.reasons, [])
        self.assertEqual(session.ranking_status, "NOT_APPLICABLE")
        self.assertNotIn("rank", [r.question_id for r in session.responses])
        self.assertNotIn("Please rank", output)
        self.assertNotIn("1. unspecified reason", output)

    def test_single_supported_factor_retains_optional_influence_without_importance(self):
        session, _, _ = converse(standard_lines("salary") + ["pull"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.candidates[0].influence, Influence.PULL)
        self.assertEqual(session.reasons, [])
        self.assertEqual(session.ranked_candidate_ids, [])
        self.assertEqual(session.ranking_response_ids, [])
        self.assertNotIn("rank", [r.question_id for r in session.responses])

    def test_multiple_classified_factors_still_follow_participant_ranking(self):
        session, _, _ = converse(["START", "INVOLUNTARY", "I was fired. The salary was too low.",
                                 "not really", *(["SKIP"] * 7), "2,1", "unknown"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual([c.subcategory for c in session.candidates], ["EMPLOYER_INITIATED", "SALARY"])
        self.assertEqual([r.subcategory for r in session.reasons], ["SALARY", "EMPLOYER_INITIATED"])
        self.assertEqual([r.role for r in session.reasons], [ReasonRole.PRIMARY, ReasonRole.CONTRIBUTING])
        self.assertEqual(session.ranked_candidate_ids, ["F2", "F1"])
        self.assertEqual(session.ranking_status, "PARTICIPANT_RANKED")
        self.assertEqual(session.reasons[0].influence, Influence.PUSH)
        self.assertEqual(session.reasons[1].influence, Influence.UNKNOWN)

    def test_more_than_three_candidates_still_require_selection_and_ranking(self):
        original = "The salary was too low. The hours were terrible. There was nowhere to advance."
        session, _, _ = converse(["START", "VOLUNTARY", original, "Another unclear factor.", "SKIP",
                                 *(["SKIP"] * 7), "4,1,2", "3,2,1"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(len(session.candidates), 4)
        self.assertEqual([r.subcategory for r in session.reasons], ["WORKING_HOURS", "SALARY", None])
        self.assertEqual(session.reasons[2].status, ClassificationStatus.UNCLASSIFIED)
        self.assertEqual(session.responses[2].text, original)
        self.assertIn("select", [r.question_id for r in session.responses])
        self.assertIn("rank", [r.question_id for r in session.responses])

    def test_employer_event_in_general_feedback_remains_feedback_only(self):
        session, _, _ = converse(["START", "INVOLUNTARY", "SKIP", "not really",
                                 "I was fired", *(["SKIP"] * 6)])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.candidates, [])
        response = next(r for r in session.responses if r.question_id == "worked_well")
        self.assertEqual(response.text, "I was fired")
        self.assertEqual(response.evidence, [])

    def test_stop_remains_available_during_optional_singleton_influence(self):
        session, _, _ = converse(["START", "INVOLUNTARY", "I was fired", "not really",
                                 *(["SKIP"] * 7), "STOP"])
        self.assertEqual(session.status, "STOPPED")
        self.assertEqual(session.candidates[0].subcategory, "EMPLOYER_INITIATED")
        self.assertEqual(session.reasons, [])


class NumericStartControlTests(unittest.TestCase):
    def test_initial_prompt_lists_numeric_controls(self):
        _, output, _ = converse(["EXIT"])
        self.assertIn("1 - Start\n0 - Exit", output)

    def test_initial_one_starts_exactly_like_text_start(self):
        text_session, _, _ = converse(standard_lines())
        self.assertEqual(text_session.status, "COMPLETED")
        for answer in ["1", " 1 "]:
            with self.subTest(answer=answer):
                lines = standard_lines()
                lines[0] = answer
                session, _, engine = converse(lines)
                self.assertEqual(engine.state, ConversationState.COMPLETED)
                expected, actual = text_session.to_dict(), session.to_dict()
                expected["session_id"] = actual["session_id"]
                expected["responses"][0]["text"] = answer
                self.assertEqual(actual, expected)

    def test_initial_zero_exits_exactly_like_text_exit(self):
        text_session, _, _ = converse(["EXIT"])
        for answer in ["0", " 0 "]:
            with self.subTest(answer=answer):
                session, output, engine = converse([answer, "must not be read"])
                self.assertEqual(engine.state, ConversationState.STOPPED)
                expected, actual = text_session.to_dict(), session.to_dict()
                expected["session_id"] = actual["session_id"]
                self.assertEqual(actual, expected)
                self.assertEqual(session.responses, [])
                self.assertIn("No more questions will be asked", output)
                self.assertNotIn(load_knowledge("questions.json")["context"], output)

    def test_text_stop_and_exit_remain_supported(self):
        for command in ["STOP", "EXIT"]:
            for prefix in [[], ["1", "1"]]:
                with self.subTest(command=command, prefix=prefix):
                    session, _, engine = converse([*prefix, command, "must not be read"])
                    self.assertEqual(session.status, "STOPPED")
                    self.assertEqual(engine.state, ConversationState.STOPPED)
                    self.assertNotIn("must not be read", [r.text for r in session.responses])

    def test_one_after_start_remains_separation_context_choice(self):
        for start in ["1", "START"]:
            with self.subTest(start=start):
                session, output, _ = converse([start, "1", "SKIP", "No", *(["SKIP"] * 7)])
                self.assertEqual(session.status, "COMPLETED")
                self.assertEqual(session.separation_context, SeparationContext.VOLUNTARY)
                self.assertEqual(session.responses[1].text, "1")
                self.assertEqual(session.responses[1].evidence, [])
                self.assertEqual(sum(r.question_id == "start" for r in session.responses), 1)
                self.assertEqual(sum(r.question_id == "context" for r in session.responses), 1)
                self.assertEqual(output.count(load_knowledge("questions.json")["start"]), 1)

    def test_numeric_ranking_with_spaces_remains_participant_controlled(self):
        for ranking, expected in [("1, 2", ["SALARY", "WORKING_HOURS"]),
                                  ("2, 1", ["WORKING_HOURS", "SALARY"])]:
            with self.subTest(ranking=ranking):
                session, _, _ = converse(["1", "1", "The salary was too low.",
                                          "The hours were terrible.", *FEEDBACK, ranking])
                self.assertEqual(session.status, "COMPLETED")
                self.assertEqual(session.ranking_status, "PARTICIPANT_RANKED")
                self.assertEqual([r.subcategory for r in session.reasons], expected)
                self.assertEqual([r.rank for r in session.reasons], [1, 2])
                response = next(r for r in session.responses if r.question_id == "rank")
                self.assertEqual(response.text, ranking)

    def test_numeric_factor_selection_remains_unchanged(self):
        reason = ("The salary was too low. The hours were terrible. "
                  "There was nowhere to advance. I had problems with my direct supervisor.")
        session, _, _ = converse(["1", "1", reason, "No", *FEEDBACK, "1, 2, 3", "3, 2, 1"])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(len(session.candidates), 4)
        self.assertEqual([r.subcategory for r in session.reasons],
                         ["CAREER_PROGRESSION", "WORKING_HOURS", "SALARY"])
        response = next(r for r in session.responses if r.question_id == "select")
        self.assertEqual(response.text, "1, 2, 3")

    def test_one_is_not_a_global_start_intent(self):
        engine = ConversationEngine(io.StringIO("1\n"), io.StringIO())
        for question_id in [None, "context", "main_reason", "select", "rank"]:
            with self.subTest(question_id=question_id):
                self.assertEqual(engine.intents.detect("1", question_id).value, "ANSWER")
        engine.state = ConversationState.EXIT_REASONS
        response = engine.ask("main_reason")
        self.assertEqual(response.text, "1")
        self.assertEqual(response.disposition, ResponseDisposition.ANSWERED)
        self.assertEqual(engine.state, ConversationState.EXIT_REASONS)

    def test_numeric_start_preserves_skip_help_and_prefer_not_to_answer(self):
        session, output, _ = converse([
            "HELP", "SKIP", "PREFER NOT TO ANSWER", "1", "1", "HELP", "SKIP",
            "PREFER NOT TO ANSWER", *(["SKIP"] * 7),
        ])
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.separation_context, SeparationContext.VOLUNTARY)
        self.assertEqual([r.text for r in session.responses[:3]],
                         ["SKIP", "PREFER NOT TO ANSWER", "1"])
        self.assertEqual(session.responses[0].disposition, ResponseDisposition.SKIPPED)
        self.assertEqual(session.responses[1].disposition, ResponseDisposition.PREFER_NOT_TO_ANSWER)
        main_response = next(r for r in session.responses if r.question_id == "main_reason")
        other_response = next(r for r in session.responses if r.question_id == "other_factors")
        self.assertEqual(main_response.disposition, ResponseDisposition.SKIPPED)
        self.assertEqual(other_response.disposition, ResponseDisposition.PREFER_NOT_TO_ANSWER)
        self.assertNotIn("HELP", [r.text for r in session.responses])
        self.assertEqual(output.count("You control what to share"), 2)
        self.assertEqual(session.clarification_attempts, {})
        self.assertEqual(session.candidates, [])

    def test_exact_zero_exits_at_later_prompts_without_numeric_choice_conflicts(self):
        reason = ("The salary was too low. The hours were terrible. "
                  "There was nowhere to advance. I had problems with my direct supervisor.")
        prefixes = {
            "context": ["1"],
            "main_reason": ["1", "1"],
            "clarification": ["1", "1", "Unclear."],
            "feedback": ["1", "1", "SKIP", "No"],
            "selection": ["1", "1", reason, "No", *FEEDBACK],
            "ranking": ["1", "1", "The salary was too low.", "The hours were terrible.", *FEEDBACK],
            "influence": ["1", "1", "salary", "No", *FEEDBACK],
        }
        for prompt, prefix in prefixes.items():
            with self.subTest(prompt=prompt):
                session, output, engine = converse([*prefix, "0", "must not be read"])
                self.assertEqual(session.status, "STOPPED")
                self.assertEqual(engine.state, ConversationState.STOPPED)
                self.assertEqual([r.text for r in session.responses], prefix)
                self.assertEqual(session.ranked_candidate_ids, [])
                self.assertIn("No more questions will be asked", output)


class SessionStateSemanticsTests(unittest.TestCase):
    def test_new_session_has_no_collected_context_or_required_ranking(self):
        session = Session()
        self.assertIsNone(session.separation_context)
        self.assertEqual(session.ranking_status, "NOT_APPLICABLE")
        summary = json.loads(json.dumps(session.to_dict()))
        self.assertIsNone(summary["separation_context"])
        self.assertEqual(summary["ranking_status"], "NOT_APPLICABLE")

    def test_initial_zero_summary_has_null_context_and_no_synthetic_responses(self):
        session, output, engine = converse(["0", "must not be read"])
        self.assertEqual(engine.state, ConversationState.STOPPED)
        expected = {
            "separation_context": None,
            "responses": [],
            "candidates": [],
            "ranked_candidate_ids": [],
            "ranking_response_ids": [],
            "ranking_status": "NOT_APPLICABLE",
            "status": "STOPPED",
            "reasons": [],
        }
        marker = load_knowledge("questions.json")["development_summary"] + "\n"
        summary = json.loads(output.split(marker, 1)[1])
        self.assertEqual({key: summary[key] for key in expected}, expected)
        self.assertEqual({key: session.to_dict()[key] for key in expected}, expected)
        self.assertIn("1 - Start\n0 - Exit", output)
        self.assertNotIn(load_knowledge("questions.json")["context"], output)

    def test_stop_exit_and_eof_before_context_collection_keep_null(self):
        for prefix in [[], ["1"], ["START"], ["1", "Not a context choice"]]:
            for command in ["0", "STOP", "EXIT", None]:
                with self.subTest(prefix=prefix, command=command):
                    lines = [*prefix, *([command] if command is not None else [])]
                    input_text = "\n".join(lines) + ("\n" if lines else "")
                    engine = ConversationEngine(io.StringIO(input_text), io.StringIO())
                    session = engine.run()
                    self.assertEqual(session.status, "STOPPED")
                    self.assertIsNone(session.separation_context)
                    self.assertEqual(session.ranking_status, "NOT_APPLICABLE")
                    self.assertEqual([r.text for r in session.responses], prefix)
                    self.assertEqual(session.candidates, [])
                    self.assertEqual(session.reasons, [])

    def test_explicit_context_decline_remains_distinct_from_null(self):
        for answer in ["PREFER_NOT_TO_ANSWER", "PREFER NOT TO ANSWER", "7"]:
            with self.subTest(answer=answer):
                session, output, _ = converse(["1", answer, "SKIP", "No", *(["SKIP"] * 7)])
                self.assertEqual(session.status, "COMPLETED")
                self.assertEqual(session.separation_context, SeparationContext.PREFER_NOT_TO_ANSWER)
                self.assertEqual(session.to_dict()["separation_context"], "PREFER_NOT_TO_ANSWER")
                self.assertEqual(session.responses[1].question_id, "context")
                self.assertEqual(session.responses[1].text, answer)
                self.assertIn(load_knowledge("questions.json")["context"], output)
                self.assertEqual(session.ranking_status, "NOT_APPLICABLE")

    def test_decline_at_start_is_not_a_separation_context_response(self):
        session, output, _ = converse(["PREFER NOT TO ANSWER", "0"])
        self.assertEqual(session.status, "STOPPED")
        self.assertIsNone(session.separation_context)
        self.assertEqual([r.question_id for r in session.responses], ["start"])
        self.assertNotIn(load_knowledge("questions.json")["context"], output)

    def test_skipped_context_stays_null_and_uses_existing_generic_question(self):
        for answer in ["SKIP", ""]:
            with self.subTest(answer=answer):
                session, output, _ = converse(["1", answer, "SKIP", "No", *(["SKIP"] * 7)])
                self.assertEqual(session.status, "COMPLETED")
                self.assertIsNone(session.separation_context)
                self.assertEqual(session.responses[1].disposition, ResponseDisposition.SKIPPED)
                self.assertIn(load_knowledge("questions.json")["main_reason"], output)
                self.assertEqual(session.ranking_status, "NOT_APPLICABLE")

    def test_pending_begins_only_when_distinct_candidates_require_ranking(self):
        engine = ConversationEngine(io.StringIO(), io.StringIO())
        session = engine.session
        self.assertEqual(session.ranking_status, "NOT_APPLICABLE")
        for text, expected_count, expected_status in [
            ("Salary was not a reason.", 0, "NOT_APPLICABLE"),
            ("The salary was too low.", 1, "NOT_APPLICABLE"),
            ("The pay was too low.", 1, "NOT_APPLICABLE"),
            ("The hours were terrible.", 2, "PENDING"),
        ]:
            with self.subTest(text=text):
                response = session.record_response("main_reason", text)
                engine._classify_response(response, reason_context=True)
                self.assertEqual(len(session.candidates), expected_count)
                self.assertEqual(session.ranking_status, expected_status)
                self.assertEqual(session.ranked_candidate_ids, [])
                self.assertEqual(session.ranking_response_ids, [])
                self.assertEqual(session.reasons, [])
        ranking = session.record_response("rank", "2, 1")
        session.rank_reasons(["F2", "F1"], [ranking.response_id])
        self.assertEqual(session.ranking_status, "PARTICIPANT_RANKED")
        repeat = session.record_response("other_factors", "The hours were terrible.")
        engine._classify_response(repeat, reason_context=True)
        self.assertEqual(session.ranking_status, "PARTICIPANT_RANKED")

    def test_interrupted_ranking_is_pending_only_when_required(self):
        four_factors = ("The salary was too low. The hours were terrible. "
                        "There was nowhere to advance. I had problems with my direct supervisor.")
        cases = [
            (["1", "1", "SKIP", "No"], 0, "NOT_APPLICABLE"),
            (["1", "1", "salary", "No", *FEEDBACK], 1, "NOT_APPLICABLE"),
            (["1", "1", "The salary was too low.", "The hours were terrible.", *FEEDBACK], 2, "PENDING"),
            (["1", "1", four_factors, "No", *FEEDBACK], 4, "PENDING"),
        ]
        for prefix, expected_count, expected_status in cases:
            for command in ["0", "STOP", "EXIT"]:
                with self.subTest(count=expected_count, command=command):
                    session, _, _ = converse([*prefix, command])
                    self.assertEqual(session.status, "STOPPED")
                    self.assertEqual(len(session.candidates), expected_count)
                    self.assertEqual(session.ranking_status, expected_status)
                    self.assertEqual(session.ranked_candidate_ids, [])
                    self.assertEqual(session.ranking_response_ids, [])
                    self.assertEqual(session.reasons, [])

    def test_completed_ranking_preserves_participant_ranked_and_skipped_states(self):
        for answer, expected_status in [("2, 1", "PARTICIPANT_RANKED"),
                                        ("SKIP", "SKIPPED"), ("PREFER NOT TO ANSWER", "SKIPPED")]:
            with self.subTest(answer=answer):
                session, _, _ = converse(standard_lines(other="The hours were terrible.") + [answer])
                self.assertEqual(session.status, "COMPLETED")
                self.assertEqual(session.ranking_status, expected_status)
                if expected_status == "PARTICIPANT_RANKED":
                    self.assertEqual(session.ranked_candidate_ids, ["F2", "F1"])
                    self.assertEqual([r.subcategory for r in session.reasons], ["WORKING_HOURS", "SALARY"])
                else:
                    self.assertEqual(session.ranked_candidate_ids, [])
                    self.assertEqual(session.reasons, [])

    def test_numeric_one_start_still_collects_explicit_context(self):
        session, _, engine = converse(["1", "1", "SKIP", "No", *(["SKIP"] * 7)])
        self.assertEqual(engine.state, ConversationState.COMPLETED)
        self.assertEqual(session.status, "COMPLETED")
        self.assertEqual(session.separation_context, SeparationContext.VOLUNTARY)
        self.assertEqual([r.text for r in session.responses[:2]], ["1", "1"])
        self.assertEqual(session.ranking_status, "NOT_APPLICABLE")
        self.assertEqual(session.candidates, [])


if __name__ == "__main__":
    unittest.main()
