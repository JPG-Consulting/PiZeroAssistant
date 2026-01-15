import unittest

from voiceassistant.speech.incremental import IncrementalSpeechCoordinator


class TestIncrementalSpeechCoordinator(unittest.TestCase):
    def test_spanish_inverted_punctuation(self) -> None:
        coordinator = IncrementalSpeechCoordinator()
        chunks = coordinator.push_delta("¿Cómo estás? ")
        chunks.extend(coordinator.push_delta("Estoy bien."))
        chunks.extend(coordinator.finish())
        self.assertEqual(chunks, ["¿Cómo estás? ", "Estoy bien."])

    def test_abbreviation_guard(self) -> None:
        coordinator = IncrementalSpeechCoordinator()
        chunks = coordinator.push_delta("El Sr. Pérez llegó.")
        chunks.extend(coordinator.finish())
        self.assertEqual(chunks, ["El Sr. Pérez llegó."])

    def test_partial_word_streaming(self) -> None:
        coordinator = IncrementalSpeechCoordinator()
        self.assertEqual(coordinator.push_delta("Hol"), [])
        chunks = coordinator.push_delta("a mundo.")
        chunks.extend(coordinator.finish())
        self.assertEqual(chunks, ["Hola mundo."])

    def test_unicode_safety(self) -> None:
        coordinator = IncrementalSpeechCoordinator()
        chunks = coordinator.push_delta("¡Buenos días, señor! ")
        chunks.extend(coordinator.push_delta("Привет мир!"))
        chunks.extend(coordinator.finish())
        self.assertEqual(chunks, ["¡Buenos días, señor! ", "Привет мир!"])

    def test_adaptive_first_chunk(self) -> None:
        coordinator = IncrementalSpeechCoordinator()
        text = "palabra " * 20
        chunks = coordinator.push_delta(text)
        self.assertEqual(len(chunks), 1)
        self.assertGreaterEqual(len(chunks[0]), 80)
        self.assertLess(len(chunks[0]), 120)

    def test_cancel_behavior(self) -> None:
        coordinator = IncrementalSpeechCoordinator()
        coordinator.cancel("user_stop")
        self.assertEqual(coordinator.push_delta("Hola."), [])
        self.assertEqual(coordinator.finish(), [])


if __name__ == "__main__":
    unittest.main()
