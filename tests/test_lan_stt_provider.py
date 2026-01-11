import unittest
from unittest.mock import Mock, patch

from voiceassistant.providers.base import ProviderError
from voiceassistant.providers.http import STTResponse
from voiceassistant.providers.lan_stt import LanHttpSTTProvider


class TestLanHttpSTTProvider(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = LanHttpSTTProvider(
            name="lan-stt",
            endpoint="http://localhost:8000",
            timeout_s=5,
            api_key=None,
        )

    @patch("voiceassistant.providers.lan_stt.requests.post")
    def test_empty_text_is_allowed(self, mock_post: Mock) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"text": ""}
        response.text = '{"text": ""}'
        mock_post.return_value = response

        result = self.provider.transcribe(b"RIFFxxxxWAVE")

        self.assertIsInstance(result, STTResponse)
        self.assertEqual(result.text, "")

    @patch("voiceassistant.providers.lan_stt.requests.post")
    def test_missing_text_key_raises(self, mock_post: Mock) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {}
        response.text = "{}"
        mock_post.return_value = response

        with self.assertRaises(ProviderError) as context:
            self.provider.transcribe(b"RIFFxxxxWAVE")

        self.assertIn(
            "STT response missing required 'text' field",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
