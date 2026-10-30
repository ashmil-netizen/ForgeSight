"""API-level tests for prediction, validation, model information, and history."""

import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx

import api.main as api_module


VALID_PAYLOAD = {
    "machine_type": "M",
    "air_temperature": 300,
    "process_temperature": 310,
    "rotational_speed": 1500,
    "torque": 40,
    "tool_wear": 100,
}


class PredictionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_directory = tempfile.TemporaryDirectory()
        api_module.DB_PATH = Path(cls.temp_directory.name) / "test_history.sqlite3"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_directory.cleanup()

    async def request_many(self, requests):
        async with api_module.app.router.lifespan_context(api_module.app):
            transport = httpx.ASGITransport(app=api_module.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                responses = []
                for method, path, payload in requests:
                    responses.append(
                        await client.request(method, path, json=payload)
                    )
                return responses

    def run_requests(self, requests):
        return asyncio.run(self.request_many(requests))

    def test_health_and_model_info(self) -> None:
        health, info = self.run_requests(
            [
                ("GET", "/api/health", None),
                ("GET", "/api/model-info", None),
            ]
        )
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(info.status_code, 200)
        self.assertEqual(info.json()["dataset_rows"], 10000)
        self.assertIn("Random Forest", info.json()["model"])
        self.assertEqual(info.json()["dataset_target_counts"], {"0": 9661, "1": 339})
        self.assertEqual(info.json()["prediction_history_count"], 0)
        self.assertEqual(
            len(info.json()["parameter_distributions"]["Air temperature"]),
            8,
        )
        self.assertIn("limitations", info.json())

    def test_prediction_is_model_generated_and_saved_to_history(self) -> None:
        response, history = self.run_requests(
            [
                ("POST", "/api/predict", VALID_PAYLOAD),
                ("GET", "/api/predictions?limit=5", None),
            ]
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertIn(result["predicted_failure"], (0, 1))
        self.assertGreaterEqual(result["failure_probability"], 0)
        self.assertLessEqual(result["failure_probability"], 1)
        self.assertEqual(result["parameters"]["Type"], "M")
        self.assertIsInstance(result["id"], int)

        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.json()[0]["id"], result["id"])
        self.assertEqual(history.json()[0]["risk_category"], result["risk_category"])

    def test_invalid_requests_are_rejected(self) -> None:
        outside_range = {**VALID_PAYLOAD, "tool_wear": 254}
        unknown_field = {**VALID_PAYLOAD, "machine_failure": 1}
        fractional_speed = {**VALID_PAYLOAD, "rotational_speed": 1500.5}
        out_of_range, unexpected_field, invalid_limit, fractional = self.run_requests(
            [
                ("POST", "/api/predict", outside_range),
                ("POST", "/api/predict", unknown_field),
                ("GET", "/api/predictions?limit=1000", None),
                ("POST", "/api/predict", fractional_speed),
            ]
        )
        self.assertEqual(out_of_range.status_code, 422)
        self.assertEqual(unexpected_field.status_code, 422)
        self.assertEqual(invalid_limit.status_code, 422)
        self.assertEqual(fractional.status_code, 422)


if __name__ == "__main__":
    unittest.main()
