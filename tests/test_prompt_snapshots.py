import unittest

from medmatch.experiments import cot, tier3_normalize


class PromptSnapshotTests(unittest.TestCase):
    def test_cot_remote_push_guidance_contains_canonical_rules(self):
        text = cot.REMOTE_EXTRACT_GUIDANCE["IV push (17)"]
        self.assertIn("per-1-mL", text)
        self.assertIn("vial solution", text)
        self.assertIn("once daily", text)

    def test_cot_local_push_guidance_contains_bid_rule(self):
        text = cot.LOCAL_EXTRACT_GUIDANCE["IV push (17)"]
        self.assertIn("daily -> once daily", text)
        self.assertIn("BID -> twice daily", text)

    def test_tier3_remote_iv_prompt_keeps_per_ml_and_ascii_rules(self):
        text = tier3_normalize.REMOTE_IV_NORMALIZE_PROMPT
        self.assertIn("per 1 mL", text)
        self.assertIn("ASCII hyphens", text)
        self.assertIn("do not add or remove fields", text.lower())

    def test_tier3_remote_oral_prompt_keeps_route_rule(self):
        text = tier3_normalize.REMOTE_ORAL_NORMALIZE_PROMPT
        self.assertIn('Always write "by mouth"', text)
        self.assertIn("Do not change numeric values", text)


if __name__ == "__main__":
    unittest.main()
