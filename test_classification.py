import unittest

from models.reasons import ClassificationStatus, Influence, ReasonEvidence
from models.session import Session, SeparationContext
from nlp.intents import Intent, IntentProcessor, normalize
from nlp.processor import Processor
from nlp.taxonomy import Taxonomy, load_knowledge


EXPECTED_SUBCATEGORIES = {
    "COMPENSATION_BENEFITS": "SALARY BONUS BENEFITS PAY_FAIRNESS FINANCIAL_RECOGNITION OTHER_COMPENSATION",
    "MANAGEMENT_LEADERSHIP": "DIRECT_SUPERVISION COMMUNICATION RECOGNITION FEEDBACK DECISION_MAKING TRUST_AUTONOMY FAIRNESS LEADERSHIP_DIRECTION OTHER_MANAGEMENT",
    "ROLE_WORK_CONTENT": "RESPONSIBILITIES ROLE_CLARITY JOB_FIT MEANINGFUL_WORK PROCESSES TOOLS_RESOURCES AUTONOMY UNCHALLENGING_WORK OTHER_ROLE",
    "CAREER_DEVELOPMENT": "PROMOTION CAREER_PROGRESSION TRAINING SKILL_DEVELOPMENT INTERNAL_MOBILITY CAREER_DIRECTION OTHER_DEVELOPMENT",
    "WORK_ENVIRONMENT_CULTURE": "TEAM_RELATIONSHIPS COLLABORATION COMMUNICATION_CLIMATE RESPECT_INCLUSION RECOGNITION_CULTURE ORGANIZATIONAL_VALUES WORKPLACE_CONFLICT PHYSICAL_ENVIRONMENT OTHER_ENVIRONMENT",
    "WORKLOAD_WORK_LIFE": "WORKLOAD WORKING_HOURS SCHEDULE FLEXIBILITY REMOTE_HYBRID COMMUTE_LOCATION STAFFING WORK_LIFE_BALANCE OTHER_WORK_CONDITION",
    "ORGANIZATIONAL_CHANGE": "RESTRUCTURING INSTABILITY STRATEGY_DIRECTION LEADERSHIP_CHANGE PROCESS_SYSTEM_CHANGE JOB_SECURITY OTHER_ORGANIZATIONAL_CHANGE",
    "PERSONAL_CIRCUMSTANCES": "RELOCATION FAMILY_CAREGIVING EDUCATION PERSONAL_OTHER",
    "EXTERNAL_OPPORTUNITY": "BETTER_COMPENSATION BETTER_ROLE CAREER_ADVANCEMENT PREFERRED_FIELD PREFERRED_EMPLOYER SELF_EMPLOYMENT OTHER_EXTERNAL_OPPORTUNITY",
    "EMPLOYMENT_EVENT": "CONTRACT_COMPLETION RETIREMENT INTERNAL_TRANSFER EMPLOYER_INITIATED MUTUAL_SEPARATION OTHER_EMPLOYMENT_EVENT",
    "OTHER": "",
    "UNCLASSIFIED": "",
}


class TaxonomyTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = Taxonomy()

    def test_taxonomy_loads(self):
        self.assertEqual(len(self.taxonomy.categories), 12)
        self.assertEqual([item["code"] for item in self.taxonomy.categories.values()],
                         [f"{i:02}" for i in range(1, 12)] + ["99"])

    def test_complete_exact_taxonomy(self):
        self.assertEqual(set(self.taxonomy.categories), set(EXPECTED_SUBCATEGORIES))
        for category, expected in EXPECTED_SUBCATEGORIES.items():
            with self.subTest(category=category):
                self.assertEqual(self.taxonomy.categories[category]["subcategories"], expected.split())

    def test_all_category_subcategory_pairs_validate(self):
        for category, item in self.taxonomy.categories.items():
            for subcategory in item["subcategories"] or [None]:
                with self.subTest(category=category, subcategory=subcategory):
                    self.assertTrue(self.taxonomy.validate(category, subcategory))

    def test_invalid_category_subcategory_pairs_are_rejected(self):
        for pair in [("UNKNOWN_CATEGORY", None), ("COMPENSATION_BENEFITS", "WORKING_HOURS"),
                     ("UNCLASSIFIED", "SALARY"), ("CAREER_DEVELOPMENT", None)]:
            with self.subTest(pair=pair):
                self.assertFalse(self.taxonomy.validate(*pair))
                with self.assertRaises(ValueError):
                    self.taxonomy.require_valid(*pair)

    def test_separation_context_values_match_model(self):
        self.assertEqual(set(self.taxonomy.separation_contexts), {value.value for value in SeparationContext})

    def test_rules_use_only_valid_taxonomy_pairs(self):
        rules = load_knowledge("classification_rules.json")
        for rule in rules["rules"] + rules["broad_factors"]:
            self.assertTrue(self.taxonomy.validate(rule["category"], rule["subcategory"]))
        self.assertEqual(len(rules["rules"]), 8)


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.processor = Processor()

    def keys(self, text, reason_context=True):
        return {item.key for item in self.processor.classify(text, reason_context).supported}

    def test_salary_classification(self):
        for text in ["The salary was too low.", "The pay wasn't competitive."]:
            with self.subTest(text=text):
                self.assertEqual(self.keys(text), {("COMPENSATION_BENEFITS", "SALARY")})

    def test_career_progression_classification(self):
        for text in ["There was nowhere for me to advance.", "I couldn't see any career growth."]:
            with self.subTest(text=text):
                self.assertEqual(self.keys(text), {("CAREER_DEVELOPMENT", "CAREER_PROGRESSION")})

    def test_working_hours_classification(self):
        for text in ["The hours were terrible.", "I was working too many hours."]:
            with self.subTest(text=text):
                self.assertEqual(self.keys(text), {("WORKLOAD_WORK_LIFE", "WORKING_HOURS")})

    def test_direct_supervision_classification(self):
        for text in ["My manager's supervision was one of the reasons.",
                     "I had problems with my direct supervisor."]:
            with self.subTest(text=text):
                self.assertEqual(self.keys(text), {("MANAGEMENT_LEADERSHIP", "DIRECT_SUPERVISION")})

    def test_relocation_classification(self):
        for text in ["I'm relocating.", "I'm moving away."]:
            with self.subTest(text=text):
                self.assertEqual(self.keys(text), {("PERSONAL_CIRCUMSTANCES", "RELOCATION")})

    def test_external_better_compensation_classification(self):
        self.assertEqual(self.keys("I accepted another job because it pays more."),
                         {("EXTERNAL_OPPORTUNITY", "BETTER_COMPENSATION")})

    def test_explicit_exclusion_prevents_false_salary_classification(self):
        text = ("Salary was good and money wasn't why I left. "
                "I left because there was nowhere to advance.")
        self.assertEqual(self.keys(text), {("CAREER_DEVELOPMENT", "CAREER_PROGRESSION")})
        result = self.processor.classify(text, reason_context=True)
        self.assertIn(("COMPENSATION_BENEFITS", "SALARY"), {item.key for item in result.excluded})

    def test_exclusion_overrides_positive_in_same_response(self):
        text = ("The salary was too low. Actually money wasn't why I left. "
                "The hours were terrible.")
        self.assertEqual(self.keys(text), {("WORKLOAD_WORK_LIFE", "WORKING_HOURS")})

    def test_one_response_can_produce_multiple_reasons(self):
        self.assertEqual(self.keys("The hours were terrible and the pay was too low."), {
            ("WORKLOAD_WORK_LIFE", "WORKING_HOURS"), ("COMPENSATION_BENEFITS", "SALARY"),
        })

    def test_keywords_alone_do_not_classify(self):
        for text in ["Salary came up in a meeting.", "My manager reviewed the schedule.",
                     "We talked about career growth.", "The hours were discussed.",
                     "The new job was mentioned.", "I read about relocation."]:
            with self.subTest(text=text):
                self.assertEqual(self.keys(text, True), set())

    def test_bare_factor_is_explicit_only_in_reason_context(self):
        self.assertEqual(self.keys("salary", False), set())
        self.assertEqual(self.keys("salary", True), {("COMPENSATION_BENEFITS", "SALARY")})
        self.assertEqual(self.processor.classify("salary", True).supported[0].influence,
                         Influence.UNKNOWN)

    def test_broad_area_does_not_add_specific_subcategory(self):
        self.assertEqual(self.keys("career growth and development", True),
                         {("CAREER_DEVELOPMENT", "OTHER_DEVELOPMENT")})

    def test_ambiguous_response_has_no_guessed_evidence(self):
        result = self.processor.classify("It was complicated.", reason_context=True)
        self.assertEqual(result.supported, [])
        self.assertEqual(result.status, ClassificationStatus.UNCLASSIFIED)

    def test_uncertain_phrases_do_not_classify(self):
        for text in ["Maybe the salary was too low.", "Perhaps the hours were terrible.",
                     "If the salary was too low, I would leave."]:
            with self.subTest(text=text):
                self.assertEqual(self.keys(text), set())

    def test_negated_claims_do_not_classify(self):
        for text in ["The salary was not too low.", "I'm not relocating.",
                     "I did not leave because the hours were terrible.",
                     "I didn't say the salary was too low.",
                     "The hours were never terrible."]:
            with self.subTest(text=text):
                self.assertEqual(self.keys(text), set())

    def test_known_negative_language_is_not_discarded(self):
        self.assertEqual(self.keys("The pay was not competitive."),
                         {("COMPENSATION_BENEFITS", "SALARY")})
        self.assertEqual(self.keys("There was no career growth."),
                         {("CAREER_DEVELOPMENT", "CAREER_PROGRESSION")})

    def test_unicode_apostrophes_and_case_normalize(self):
        self.assertEqual(self.keys("THE PAY WASN’T COMPETITIVE."),
                         {("COMPENSATION_BENEFITS", "SALARY")})
        self.assertEqual(self.keys("I’m moving away."), {("PERSONAL_CIRCUMSTANCES", "RELOCATION")})

    def test_push_pull_is_assigned_only_when_supported(self):
        self.assertEqual(self.processor.classify("The salary was too low.", reason_context=True).supported[0].influence,
                         Influence.PUSH)
        self.assertEqual(self.processor.classify(
            "I accepted another job because it pays more.", reason_context=True).supported[0].influence, Influence.PULL)
        self.assertEqual(self.processor.classify("I'm relocating.", reason_context=True).supported[0].influence,
                         Influence.UNKNOWN)
        self.assertEqual(self.processor.classify("salary", True).supported[0].influence,
                         Influence.UNKNOWN)

    def test_external_topic_does_not_create_push_or_pull_guess(self):
        result = self.processor.classify("At another job the salary was too low.", reason_context=True)
        self.assertEqual(result.supported[0].influence, Influence.UNKNOWN)

    def test_personal_causation_does_not_imply_employment_push(self):
        for text in ["I left because I'm relocating.", "I left because of personal reasons."]:
            with self.subTest(text=text):
                result = self.processor.classify(text, True)
                self.assertEqual(result.supported[0].influence, Influence.UNKNOWN)

    def test_personal_reasons_are_accepted(self):
        for text in ["personal reasons", "personal matters", "I left for personal reasons.",
                     "My reasons are personal."]:
            with self.subTest(text=text):
                self.assertEqual(self.keys(text, True),
                                 {("PERSONAL_CIRCUMSTANCES", "PERSONAL_OTHER")})

    def test_personal_topic_alone_is_not_a_reason(self):
        for text in ["We discussed personal reasons in a workshop.",
                     "Someone mentioned personal circumstances."]:
            with self.subTest(text=text):
                self.assertEqual(self.keys(text, True), set())

    def test_classifier_has_no_ranking_fields_or_scores(self):
        evidence = self.processor.classify("The hours were terrible.", reason_context=True).supported[0]
        for field in ["rank", "role", "score", "probability", "confidence", "honesty", "emotion"]:
            self.assertFalse(hasattr(evidence, field))

    def test_original_free_text_remains_unchanged(self):
        original = "  The PAY wasn’t competitive!  "
        session = Session()
        response = session.record_response("main_reason", original)
        session.record_evidence(response, self.processor.classify(original, reason_context=True).evidence)
        self.assertEqual(session.responses[0].text, original)
        self.assertNotEqual(normalize(original), original)

    def test_contradictory_responses_become_conflicting(self):
        session = Session()
        first = session.record_response("main_reason", "Salary was one of my reasons.")
        session.record_evidence(first, self.processor.classify(first.text, reason_context=True).evidence)
        second = session.record_response("other_factors", "Actually salary wasn't really why I left.")
        session.record_evidence(second, self.processor.classify(second.text, reason_context=True).evidence)
        candidate = session.candidates[0]
        self.assertEqual(candidate.status, ClassificationStatus.CONFLICTING)
        self.assertEqual(candidate.supporting_response_ids, [first.response_id])
        self.assertEqual(candidate.excluding_response_ids, [second.response_id])
        self.assertEqual(set(candidate.source_response_ids), {first.response_id, second.response_id})

    def test_exclusion_then_positive_also_preserves_conflict(self):
        session = Session()
        for text in ["Salary was not a reason.", "The salary was too low."]:
            response = session.record_response("reason", text)
            session.record_evidence(response, self.processor.classify(text, reason_context=True).evidence)
        self.assertEqual(session.candidates[0].status, ClassificationStatus.CONFLICTING)
        self.assertEqual(session.candidates[0].excluding_response_ids, ["R1"])

    def test_exclusion_alone_does_not_create_candidate(self):
        session = Session()
        response = session.record_response("reason", "Salary wasn't why I left.")
        session.record_evidence(response, self.processor.classify(response.text, reason_context=True).evidence)
        self.assertEqual(session.candidates, [])
        self.assertFalse(response.evidence[0].supported)

    def test_repeated_factor_merges_sources_without_losing_responses(self):
        session = Session()
        for text in ["The salary was too low.", "The pay wasn't competitive."]:
            response = session.record_response("reason", text)
            session.record_evidence(response, self.processor.classify(text, reason_context=True).evidence)
        self.assertEqual(len(session.candidates), 1)
        self.assertEqual(session.candidates[0].source_response_ids, ["R1", "R2"])
        self.assertEqual(len(session.responses), 2)

    def test_domain_model_enforces_maximum_three_ranked_reasons(self):
        session = Session()
        response = session.record_response("reason", "four voluntary factors")
        for index, category in enumerate(["OTHER", "UNCLASSIFIED", "PERSONAL_CIRCUMSTANCES",
                                          "EXTERNAL_OPPORTUNITY"]):
            subcategory = [None, None, "PERSONAL_OTHER", "OTHER_EXTERNAL_OPPORTUNITY"][index]
            session.record_evidence(response, [ReasonEvidence(category, subcategory, category)])
        ranking = session.record_response("rank", "1,2,3,4")
        with self.assertRaises(ValueError):
            session.rank_reasons([c.candidate_id for c in session.candidates], [ranking.response_id])
        self.assertEqual(session.reasons, [])

    def test_domain_ranking_rejects_duplicates_unknown_ids_and_missing_reference(self):
        session = Session()
        response = session.record_response("reason", "The salary was too low.")
        session.record_evidence(response, self.processor.classify(response.text, reason_context=True).evidence)
        for ids, references in [(["F1", "F1"], ["R1"]), (["F9"], ["R1"]),
                                (["F1"], []), ([], ["R1"])]:
            with self.subTest(ids=ids, references=references), self.assertRaises(ValueError):
                session.rank_reasons(ids, references)


class IntentTests(unittest.TestCase):
    def test_commands_are_exact_not_substring_matches(self):
        processor = IntentProcessor()
        self.assertEqual(processor.detect("HELP!"), Intent.HELP)
        self.assertEqual(processor.detect("  prefer not to answer  "), Intent.PREFER_NOT_TO_ANSWER)
        self.assertEqual(processor.detect("My supervisor did not help."), Intent.ANSWER)
        self.assertEqual(processor.detect("I wanted to stop working late."), Intent.ANSWER)

    def test_no_more_factors_intent_requires_additional_factor_context(self):
        processor = IntentProcessor()
        for answer in ["no", "not really", "none", "nothing else"]:
            with self.subTest(answer=answer):
                self.assertEqual(processor.detect(answer, "other_factors"), Intent.NO_MORE_FACTORS)
                self.assertEqual(processor.detect(answer, "main_reason"), Intent.ANSWER)
                self.assertEqual(processor.detect(answer, "recommend"), Intent.ANSWER)
                self.assertEqual(processor.detect(answer), Intent.ANSWER)
                self.assertFalse(processor.has_no_more_factors(answer))
        self.assertEqual(processor.detect("no other reasons except salary", "other_factors"),
                         Intent.ANSWER)

    def test_career_ambiguity_is_not_an_automatic_classification(self):
        processor = Processor()
        self.assertEqual(processor.classify("new career opportunities", True).supported, [])
        self.assertIsNone(processor.clarification_for("new career opportunities"))
        self.assertIsNotNone(processor.clarification_for("new career opportunities", True))


class FeedbackContextClassificationTests(unittest.TestCase):
    def setUp(self):
        self.processor = Processor()

    def assert_feedback_only(self, text):
        result = self.processor.classify(text, reason_context=False)
        self.assertEqual(result.supported, [])
        self.assertEqual(result.evidence, [])
        session = Session()
        response = session.record_response("wish_different", text)
        session.record_evidence(response, result.evidence)
        self.assertEqual(session.candidates, [])
        self.assertEqual(session.reasons, [])
        self.assertEqual(response.text, text)

    def test_strong_salary_in_feedback_has_no_exit_reason(self):
        self.assert_feedback_only("The salary was too low.")

    def test_strong_salary_in_reason_context_still_classifies(self):
        result = self.processor.classify("The salary was too low.", reason_context=True)
        self.assertEqual({item.key for item in result.supported},
                         {("COMPENSATION_BENEFITS", "SALARY")})

    def test_strong_hours_in_feedback_has_no_exit_reason(self):
        self.assert_feedback_only("The hours were terrible.")

    def test_strong_hours_in_reason_context_still_classifies(self):
        result = self.processor.classify("The hours were terrible.", reason_context=True)
        self.assertEqual({item.key for item in result.supported},
                         {("WORKLOAD_WORK_LIFE", "WORKING_HOURS")})

    def test_strong_supervision_in_feedback_has_no_exit_reason(self):
        for text in ["My manager's supervision was one of the reasons.",
                     "I had problems with my direct supervisor.", "My manager was excellent."]:
            with self.subTest(text=text):
                self.assert_feedback_only(text)

    def test_strong_career_progression_in_feedback_has_no_exit_reason(self):
        for text in ["There was nowhere for me to advance.", "I couldn't see any career growth."]:
            with self.subTest(text=text):
                self.assert_feedback_only(text)

    def test_all_initial_specific_rules_require_reason_context(self):
        examples = [
            ("The salary was too low.", "COMPENSATION_BENEFITS", "SALARY"),
            ("There was nowhere for me to advance.", "CAREER_DEVELOPMENT", "CAREER_PROGRESSION"),
            ("The hours were terrible.", "WORKLOAD_WORK_LIFE", "WORKING_HOURS"),
            ("I had problems with my direct supervisor.", "MANAGEMENT_LEADERSHIP", "DIRECT_SUPERVISION"),
            ("I'm relocating.", "PERSONAL_CIRCUMSTANCES", "RELOCATION"),
            ("I accepted another job because it pays more.", "EXTERNAL_OPPORTUNITY", "BETTER_COMPENSATION"),
            ("I left for personal reasons.", "PERSONAL_CIRCUMSTANCES", "PERSONAL_OTHER"),
        ]
        for text, category, subcategory in examples:
            with self.subTest(text=text):
                self.assert_feedback_only(text)
                result = self.processor.classify(text, reason_context=True)
                self.assertEqual({item.key for item in result.supported}, {(category, subcategory)})

    def test_multiple_strong_factors_are_feedback_only_and_text_is_unchanged(self):
        self.assert_feedback_only("  The PAY wasn’t competitive! The hours were terrible.  ")


class SeparationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.processor = Processor()

    def test_explicit_employer_initiated_statements_classify_in_reason_context(self):
        for text in ["I was fired", "I was terminated", "The company terminated my employment"]:
            with self.subTest(text=text):
                result = self.processor.classify(text, reason_context=True)
                self.assertEqual({item.key for item in result.supported},
                                 {("EMPLOYMENT_EVENT", "EMPLOYER_INITIATED")})
                self.assertEqual(result.supported[0].influence, Influence.UNKNOWN)

    def test_employer_initiated_language_stays_feedback_only_without_reason_context(self):
        for text in ["I was fired", "I was terminated", "The company terminated my employment"]:
            with self.subTest(text=text):
                self.assertEqual(self.processor.classify(text, reason_context=False).evidence, [])

    def test_fired_does_not_infer_a_termination_cause(self):
        result = self.processor.classify("I was fired", reason_context=True)
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.supported[0].key, ("EMPLOYMENT_EVENT", "EMPLOYER_INITIATED"))
        self.assertFalse(any(hasattr(result.supported[0], field)
                             for field in ["cause", "guilt", "criminal", "misconduct", "score"]))

    def test_employer_initiated_rule_does_not_use_topic_mentions_or_negated_claims(self):
        for text in ["We discussed someone being fired.", "I was not fired.",
                     "I didn't say I was fired.", "Maybe I was terminated.",
                     "I was fired up about the work.", "fired", "terminated", "INVOLUNTARY",
                     "Someone said I was fired.", "I was never terminated.",
                     "The company did not terminate my employment."]:
            with self.subTest(text=text):
                self.assertEqual(self.processor.classify(text, reason_context=True).supported, [])

    def test_volunteered_misconduct_is_unsupported_not_a_finding(self):
        for text in ["I cursed at someone", "I robbed the company", "I was a thief", "I broke company rules"]:
            with self.subTest(text=text):
                result = self.processor.classify(text, reason_context=True)
                self.assertEqual(result.evidence, [])
                self.assertEqual(result.status, ClassificationStatus.UNCLASSIFIED)

    def test_conduct_responses_preserve_original_text_and_unclassified_sources(self):
        for text in ["  I cursed at someone  ", "I robbed the company", "I was a thief", "I broke company rules"]:
            with self.subTest(text=text):
                session = Session()
                response = session.record_response("main_reason", text)
                session.record_evidence(response, self.processor.classify(text, True).evidence)
                session.mark_unclassified([response.response_id])
                self.assertEqual(response.text, text)
                self.assertEqual(session.to_dict()["responses"][0]["text"], text)
                self.assertEqual(len(session.candidates), 1)
                candidate = session.candidates[0]
                self.assertEqual(candidate.key, ("UNCLASSIFIED", None))
                self.assertEqual(candidate.influence, Influence.UNKNOWN)
                self.assertEqual(candidate.source_response_ids, [response.response_id])
                self.assertEqual(session.reasons, [])

    def test_employer_event_contradiction_retains_both_sources_without_cause(self):
        session = Session()
        for question_id, text in [("main_reason", "I was fired"),
                                  ("other_factors", "Being fired was not a reason")]:
            response = session.record_response(question_id, text)
            session.record_evidence(response, self.processor.classify(text, True).evidence)
        self.assertEqual(len(session.candidates), 1)
        candidate = session.candidates[0]
        self.assertEqual(candidate.key, ("EMPLOYMENT_EVENT", "EMPLOYER_INITIATED"))
        self.assertEqual(candidate.status, ClassificationStatus.CONFLICTING)
        self.assertEqual(candidate.supporting_response_ids, ["R1"])
        self.assertEqual(candidate.excluding_response_ids, ["R2"])
        self.assertEqual(candidate.source_response_ids, ["R1", "R2"])
        self.assertEqual(candidate.influence, Influence.UNKNOWN)

    def test_separate_unclassified_origins_have_separate_candidates(self):
        session = Session()
        first = session.record_response("main_reason", "One circumstance is difficult to explain.")
        second = session.record_response("other_factors", "An unrelated circumstance mattered too.")
        session.mark_unclassified([first.response_id])
        session.mark_unclassified([second.response_id])
        self.assertEqual(len(session.candidates), 2)
        self.assertNotEqual(session.candidates[0].candidate_id, session.candidates[1].candidate_id)
        self.assertEqual([c.source_response_ids for c in session.candidates], [["R1"], ["R2"]])
        self.assertEqual([c.status for c in session.candidates],
                         [ClassificationStatus.UNCLASSIFIED, ClassificationStatus.UNCLASSIFIED])

    def test_clarifications_share_only_their_own_unclassified_origin(self):
        session = Session()
        first = session.record_response("main_reason", "First unclear circumstance.")
        clarification = session.record_response("clarify_first", "More about the first circumstance.")
        second = session.record_response("other_factors", "Second unclear circumstance.")
        session.mark_unclassified([first.response_id, clarification.response_id])
        session.mark_unclassified([second.response_id])
        further = session.record_response("clarify_second", "Further explanation of the first.")
        session.mark_unclassified([first.response_id, clarification.response_id, further.response_id])
        self.assertEqual(len(session.candidates), 2)
        self.assertEqual(session.candidates[0].source_response_ids, ["R2", "R1", "R4"])
        self.assertEqual(session.candidates[1].source_response_ids, ["R3"])
        self.assertEqual(session.candidates[0].supporting_response_ids, ["R1"])

    def test_classified_employment_event_and_unresolved_factor_remain_distinct(self):
        session = Session()
        first = session.record_response("main_reason", "I was fired.")
        session.record_evidence(first, self.processor.classify(first.text, True).evidence)
        second = session.record_response("other_factors", "Something else is difficult to describe.")
        session.mark_unclassified([second.response_id])
        self.assertEqual([c.category for c in session.candidates], ["EMPLOYMENT_EVENT", "UNCLASSIFIED"])
        self.assertEqual([c.source_response_ids for c in session.candidates], [["R1"], ["R2"]])

    def test_identical_unresolved_text_at_different_origins_is_not_silently_merged(self):
        session = Session()
        for question_id in ["main_reason", "other_factors"]:
            response = session.record_response(question_id, "Hard to explain.")
            session.mark_unclassified([response.response_id])
        self.assertEqual(len(session.candidates), 2)
        self.assertEqual([c.supporting_response_ids for c in session.candidates], [["R1"], ["R2"]])


if __name__ == "__main__":
    unittest.main()
