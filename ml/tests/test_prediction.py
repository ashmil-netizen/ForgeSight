"""Unit tests for prediction validation, risk bands, and inference."""

import unittest

from prediction import PredictionService


VALID_INPUT = {
    "Type": "M",
    "Air temperature": 300.0,
    "Process temperature": 310.0,
    "Rotational speed": 1500,
    "Torque": 40.0,
    "Tool wear": 100,
}


class PredictionServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = PredictionService()

    def test_prediction_contains_model_output_and_guidance(self) -> None:
        result = self.service.predict(VALID_INPUT)
        self.assertIn(result["predicted_failure"], (0, 1))
        self.assertGreaterEqual(result["failure_probability"], 0)
        self.assertLessEqual(result["failure_probability"], 1)
        self.assertIn(result["risk_category"], {"Low", "Medium", "High"})
        self.assertTrue(result["recommendations"])
        self.assertIn("educational prediction", result["probability_note"])

    def test_detects_held_out_failure_example_with_high_score(self) -> None:
        held_out_failure = {
            "Type": "L",
            "Air temperature": 298.9,
            "Process temperature": 310.2,
            "Rotational speed": 2737,
            "Torque": 8.8,
            "Tool wear": 142,
        }
        result = self.service.predict(held_out_failure)
        self.assertEqual(result["predicted_failure"], 1)
        self.assertGreater(result["failure_probability"], 0.5)
        self.assertEqual(result["risk_category"], "High")

    def test_risk_boundaries_are_documented(self) -> None:
        self.assertEqual(self.service.risk_category(0.0999), "Low")
        self.assertEqual(self.service.risk_category(0.10), "Medium")
        self.assertEqual(self.service.risk_category(0.2999), "Medium")
        self.assertEqual(self.service.risk_category(0.30), "High")

    def test_rejects_out_of_benchmark_range_and_nonfinite_values(self) -> None:
        out_of_range = {**VALID_INPUT, "Tool wear": 254}
        with self.assertRaisesRegex(ValueError, "AI4I dataset range"):
            self.service.predict(out_of_range)

        nonfinite = {**VALID_INPUT, "Torque": float("inf")}
        with self.assertRaisesRegex(ValueError, "finite number"):
            self.service.predict(nonfinite)

    def test_rejects_unknown_machine_type(self) -> None:
        invalid_type = {**VALID_INPUT, "Type": "X"}
        with self.assertRaisesRegex(ValueError, "one of L, M, or H"):
            self.service.predict(invalid_type)

    def test_integer_measurements_and_rule_recommendations(self) -> None:
        fractional_speed = {**VALID_INPUT, "Rotational speed": 1500.5}
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            self.service.predict(fractional_speed)

        high_wear = {**VALID_INPUT, "Tool wear": 220}
        recommendations = self.service.predict(high_wear)["recommendations"]
        self.assertTrue(any("Tool wear is at least 200 minutes" in item for item in recommendations))


if __name__ == "__main__":
    unittest.main()
