# Changelog

## Unreleased

### Fixed
- Prevented LLM and TTS invocation when STT returns an empty transcript.
- Clarified LAN STT provider contract: empty text is valid; missing text is an error.
- Improved ProviderError message consistency for STT failures.
- Avoided spending LLM tokens on silence by short-circuiting empty transcripts.

### Added
- Tests ensuring empty STT transcripts short-circuit the pipeline safely.
- Debug-level logging for empty transcript detection (non-noisy).
