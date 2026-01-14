import unittest
from unittest.mock import Mock, patch

from voiceassistant.providers.base import LLMRequest, ProviderError
from voiceassistant.providers.openai_llm import OpenAILLMProvider


class TestOpenAILLMProvider(unittest.TestCase):
    def setUp(self) -> None:
        self.endpoint = "https://api.openai.com/v1/chat/completions"
        self.provider = OpenAILLMProvider(
            name="openai",
            endpoint=self.endpoint,
            timeout_s=5,
            api_key="test-key",
            model="gpt-test",
        )

    @patch("voiceassistant.providers.openai_llm.requests.post")
    def test_complete_uses_endpoint_and_payload(self, mock_post: Mock) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"choices": [{"message": {"content": "hello"}}]}
        mock_post.return_value = response

        req = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="be helpful",
            max_tokens=42,
        )

        result = self.provider.complete(req)

        self.assertEqual(result.text, "hello")
        self.assertEqual(mock_post.call_args.args[0], self.endpoint)
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "gpt-test")
        self.assertEqual(payload["max_tokens"], 42)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "be helpful")
        self.assertEqual(payload["messages"][1]["role"], "user")

    @patch("voiceassistant.providers.openai_llm.requests.post")
    def test_missing_content_raises(self, mock_post: Mock) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"choices": [{"message": {}}]}
        mock_post.return_value = response

        req = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="be helpful",
        )

        with self.assertRaises(ProviderError):
            self.provider.complete(req)


if __name__ == "__main__":
    unittest.main()
