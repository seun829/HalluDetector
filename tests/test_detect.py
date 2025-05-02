import unittest
from src.hallu_detector.detect import is_hallucinated

class TestHallucinationDetection(unittest.TestCase):
    def test_exact_match(self):
        # Correct answer should not be flagged as hallucination
        self.assertFalse(is_hallucinated("Paris", "Paris"))

    def test_case_insensitive_match(self):
        # Matching should ignore case and whitespace
        self.assertFalse(is_hallucinated("  paris  ", "Paris"))

    def test_simple_mismatch(self):
        # Different answers should be flagged
        self.assertTrue(is_hallucinated("London", "Paris"))

    def test_partial_match(self):
        # Partial or extra info can still count as hallucination
        self.assertTrue(is_hallucinated("Paris, France", "Paris"))

    def test_empty_strings(self):
        # Empty response should be treated as hallucination if truth is non-empty
        self.assertTrue(is_hallucinated("", "Paris"))
        # Both empty → not hallucinated (no claim made)
        self.assertFalse(is_hallucinated("", ""))

if __name__ == "__main__":
    unittest.main()

