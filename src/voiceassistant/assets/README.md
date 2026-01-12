# Assets

This directory groups runtime assets by semantic ownership.

- `assets/ux/**` is reserved for UX-owned assets only.
- The state machine must never load assets directly.
- Providers and playback must remain asset-agnostic.
