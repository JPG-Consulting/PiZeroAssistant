import unittest
from unittest.mock import Mock, patch

from voiceassistant.providers.base import ProviderError
from voiceassistant.providers.openai_tts import OpenAITTSProvider


class TestOpenAITTSProvider(unittest.TestCase):
    def setUp(self) -> None:
        self.endpoint = "https://api.openai.com/v1/audio/speech"
        self.provider = OpenAITTSProvider(
            name="openai",
            endpoint=self.endpoint,
            timeout_s=5,
            api_key="test-key",
            model="tts-model",
            voice="alloy",
        )

    @patch("voiceassistant.providers.openai_tts.requests.post")
    def test_synthesize_uses_endpoint_and_payload(self, mock_post: Mock) -> None:
        response = Mock()
        response.status_code = 200
        response.headers = {"Content-Type": "audio/mpeg"}
        response.iter_content.return_value = iter([b"data"])
        mock_post.return_value = response

        result = self.provider.synthesize("Hello world")

        self.assertIsNotNone(result.audio)
        self.assertEqual(mock_post.call_args.args[0], self.endpoint)
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "tts-model")
        self.assertEqual(payload["input"], "Hello world")
        self.assertEqual(payload["voice"], "alloy")
        headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertTrue(mock_post.call_args.kwargs["stream"])

    @patch("voiceassistant.providers.openai_tts.requests.post")
    def test_voice_optional(self, mock_post: Mock) -> None:
        provider = OpenAITTSProvider(
            name="openai",
            endpoint=self.endpoint,
            timeout_s=5,
            api_key="test-key",
            model="tts-model",
            voice=None,
        )
        response = Mock()
        response.status_code = 200
        response.headers = {"Content-Type": "audio/mpeg"}
        response.iter_content.return_value = iter([b"data"])
        mock_post.return_value = response

        provider.synthesize("Hello world")

        payload = mock_post.call_args.kwargs["json"]
        self.assertNotIn("voice", payload)

    @patch("voiceassistant.providers.openai_tts.requests.post")
    def test_non_200_raises(self, mock_post: Mock) -> None:
        response = Mock()
        response.status_code = 400
        response.text = "bad request"
        response.close = Mock()
        mock_post.return_value = response

        with self.assertRaises(ProviderError):
            self.provider.synthesize("Hello world")

        response.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
