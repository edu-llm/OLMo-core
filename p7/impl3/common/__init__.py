"""Shared building blocks for the P7 tutor-layer post-training implementations.

Everything that is common across Implementations 2/3/4 (chat formatting +
assistant-only loss masking, per-dialogue System-Instruction generation, model
loading, the SFT training core, forward-KL measurement, and the Impl-3 per-token
weighting) lives here so each implementation folder stays a thin, readable
entrypoint.
"""
