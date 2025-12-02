---
profile:
  name: dev-openai
  version: 1.0.0
  description: Custom OpenAI-focused development profile
  extends: developer-expertise:dev

providers:
  - module: provider-openai
    config:
      debug: true
      raw_debug: true
      default_model: gpt-5.1-codex
---
