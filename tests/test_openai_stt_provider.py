import unittest
from unittest.mock import Mock, patch

from voiceassistant.providers.base import ProviderError
from voiceassistant.providers.openai_stt import OpenAISTTProvider


class TestOpenAISTTProvider(unittest.TestCase):
    def setUp(self) -> None:
        self.endpoint = "https://api.openai.com/v1/audio/transcriptions"
        self.provider = OpenAISTTProvider(
            name="openai-stt",
            endpoint=self.endpoint,
            timeout_s=5,
            api_key="test-key",
            model="whisper-1",
        )
        self.wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt "

    @patch("voiceassistant.providers.openai_stt.requests.post")
    def test_transcribe_uses_endpoint_headers_and_payload(self, mock_post: Mock) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"text": "hello"}
        mock_post.return_value = response

        result = self.provider.transcribe(self.wav_bytes)

        self.assertEqual(result.text, "hello")
        self.assertEqual(mock_post.call_args.args[0], self.endpoint)
        headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        files = mock_post.call_args.kwargs["files"]
        self.assertIn("file", files)
        filename, content, content_type = files["file"]
        self.assertEqual(filename, "audio.wav")
        self.assertEqual(content, self.wav_bytes)
        self.assertEqual(content_type, "audio/wav")
        data = mock_post.call_args.kwargs["data"]
        self.assertEqual(data["model"], "whisper-1")

    @patch("voiceassistant.providers.openai_stt.requests.post")
    def test_non_200_raises(self, mock_post: Mock) -> None:
        response = Mock()
        response.status_code = 500
        mock_post.return_value = response

        with self.assertRaises(ProviderError):
            self.provider.transcribe(self.wav_bytes)


if __name__ == "__main__":
    unittest.main()
