
import unittest
from unittest.mock import patch
import os

# Ensure environment variable is present
os.environ["OPENWEATHER_API_KEY"] = os.getenv("OPENWEATHER_API_KEY")

class TestLangGraphPipeline(unittest.TestCase):

    def test_router_weather(self):
        input_state = {"query": "What's the weather today?"}
        output = router_node(input_state)
        self.assertEqual(output["route"], "weather")

    def test_router_document(self):
        input_state = {"query": "Tell me about LangChain."}
        output = router_node(input_state)
        self.assertEqual(output["route"], "document")

    @patch("requests.get")
    def test_weather_node(self, mock_get):
        mock_get.return_value.json.return_value = {
            "main": {"temp": 28.7},
            "weather": [{"description": "sunny"}]
        }
        state = {"query": "What is the temperature in Bangalore?"}
        result = weather_node(state)
        self.assertIn("temperature", result["weather_data"].lower())
        self.assertIn("°c", result["weather_data"].lower())

    def test_rag_node_response(self):
        state = {"query": "What is LangChain used for?"}
        result = rag_node(state)
        self.assertIsInstance(result["document_answer"], str)
        self.assertGreater(len(result["document_answer"]), 5)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
