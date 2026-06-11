import base64
import logging
import re
import os


# [SecOpsAgent]: Canary Token Definition
CANARY_TOKEN = "e8f3a9-SECRET-DO-NOT-SHARE"

# Configure logging for security audit
audit_log_path = os.path.join("data", "security_audit.log")
logging.basicConfig(
    filename=audit_log_path,
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)


class SafetyGuard:
    """Input/Output Guardrail layer for checking safety and policy compliance."""

    def __init__(self) -> None:
        # [ThreatIntelAgent]: Threat models, heuristics, and regex signatures.

        # Blocklist keywords (checked against raw and normalized inputs)
        self._malicious_keywords = {
            "hack",
            "exploit",
            "ddos",
            "malware",
            "ransomware",
            "trojan",
            "bomb",
            "explosive",
            "weapon",
            "steal",
            "illegal",
            "pirate",
            "bypass",
            "jailbreak",
        }

        # Toxic / unsafe output patterns
        self._toxic_keywords = {
            "hate speech",
            "kill yourself",
            "idiot",
            "stupid",
            "slur",
            "fuck",
            "shit",
            "bastard",
        }

        # Regex for instruction override / system prompt leak
        self._instruction_bypass_regex = re.compile(
            r"(ignore\s+(all\s+)?previous\s+instructions|disregard\s+all\s+instructions|system\s+override|override\s+system\s+prompt|reveal\s+(your\s+)?system\s+prompt)",
            re.IGNORECASE,
        )

        # Regex for roleplay, simulated environment, and persona adoption bypasses
        self._roleplay_bypass_regex = re.compile(
            r"(dan\s+mode|do\s+anything\s+now|developer\s+mode|act\s+as\s+a\s+free\s+ai|pretend\s+you\s+are|hypothetical\s+scenario|simulate\s+a\s+developer|you\s+are\s+now\s+in\s+developer)",
            re.IGNORECASE,
        )

        # Base64 pattern detection regex (looks for continuous base64 characters)
        self._b64_regex = re.compile(r"\b[A-Za-z0-9+/]{12,}=*\b")

    def _normalize_text(self, text: str) -> str:
        """Removes all non-alphanumeric characters to defeat punctuation-based obfuscation."""
        return re.sub(r"[^a-zA-Z0-9]", "", text).lower()

    def _decode_base64_strings(self, text: str) -> list[str]:
        """Detects and decodes base64-like substrings to counter encoded prompt injections."""
        decoded_strings = []
        for match in self._b64_regex.finditer(text):
            candidate = match.group(0)
            try:
                # Add padding if needed
                missing_padding = len(candidate) % 4
                if missing_padding:
                    candidate += "=" * (4 - missing_padding)
                decoded_bytes = base64.b64decode(candidate)
                decoded_text = decoded_bytes.decode("utf-8", errors="ignore").strip()
                # Ensure the result is primarily readable text to avoid false positives on random hashes
                if len(decoded_text) > 4 and all(
                    c.isprintable() or c.isspace() for c in decoded_text
                ):
                    decoded_strings.append(decoded_text)
            except Exception:
                pass
        return decoded_strings

    def _check_keywords(self, text: str) -> tuple[bool, str]:
        """Scans the text for malicious blocklisted keywords.

        Checks both the raw input (word-bounded) and a fully normalized version.
        """
        text_lower = text.lower()
        normalized = self._normalize_text(text)

        # 1. Check raw text
        for keyword in self._malicious_keywords:
            raw_pattern = rf"\b{re.escape(keyword)}s?\b"
            if re.search(raw_pattern, text_lower):
                return False, f"Malicious keyword detected: '{keyword}'."

            # 2. Substring check on normalized to defeat obfuscated variations (e.g. h.a.c.k)
            if keyword in normalized:
                return False, f"Obfuscated malicious keyword detected: '{keyword}'."

        return True, "OK"

    def _check_regex_patterns(self, text: str) -> tuple[bool, str]:
        """Scans the text for complex injection and instruction-override regex structures."""
        if self._instruction_bypass_regex.search(text):
            return False, "Instruction override attempt detected."

        if self._roleplay_bypass_regex.search(text):
            return False, "Roleplay/Persona jailbreak attempt detected."

        return True, "OK"

    def _check_heuristics(self, text: str) -> tuple[bool, str]:
        """Applies advanced heuristics, such as decoding base64 payloads and evaluating character distributions."""
        # 1. Base64 payload decoding defense
        decoded_payloads = self._decode_base64_strings(text)
        for payload in decoded_payloads:
            # Recursively run keyword and regex checks on decoded content
            kw_ok, kw_msg = self._check_keywords(payload)
            if not kw_ok:
                return False, f"Encoded payload violation: {kw_msg}"

            rx_ok, rx_msg = self._check_regex_patterns(payload)
            if not rx_ok:
                return False, f"Encoded payload violation: {rx_msg}"

        # 2. Excessive special character check (obfuscation heuristic)
        if len(text) > 10:
            alphanumeric_count = sum(c.isalnum() for c in text)
            spaces_count = sum(c.isspace() for c in text)
            meaningful_len = len(text) - spaces_count
            if meaningful_len > 10:
                special_ratio = (meaningful_len - alphanumeric_count) / meaningful_len
                if special_ratio > 0.45:
                    return (
                        False,
                        "Excessive special characters/obfuscation pattern detected.",
                    )

        return True, "OK"

    def check_input(self, prompt: str) -> tuple[bool, str]:
        """Evaluates whether the user prompt violates safety and policy guidelines.

        If a violation is detected, a WARNING is logged.

        Args:
            prompt (str): The user\'s input prompt.

        Returns:
            tuple[bool, str]: (True, "OK") if safe, (False, rejection_reason) if unsafe.
        """
        if not prompt or not isinstance(prompt, str):
            logging.warning(
                "InputCheck: Rejected - Input must be a non-empty string. Offending Text: ",
                prompt,
            )
            return False, "Input must be a non-empty string."

        # Modular check pipeline
        ok, msg = self._check_keywords(prompt)
        if not ok:
            logging.warning(f"InputCheck: Blocked - {msg}. Offending Text: '{prompt}'")
            return False, msg

        ok, msg = self._check_regex_patterns(prompt)
        if not ok:
            logging.warning(f"InputCheck: Blocked - {msg}. Offending Text: '{prompt}'")
            return False, msg

        ok, msg = self._check_heuristics(prompt)
        if not ok:
            logging.warning(f"InputCheck: Blocked - {msg}. Offending Text: '{prompt}'")
            return False, msg

        return True, "OK"

    def check_output(self, response: str) -> tuple[bool, str]:
        """Evaluates whether the generated response violates safety or policy guidelines.

        If a violation is detected, a WARNING is logged.

        Args:
            response (str): The model\'s generated response.

        Returns:
            tuple[bool, str]: (True, "OK") if safe, (False, fallback_message) if unsafe.
        """
        if not response or not isinstance(response, str):
            logging.warning(
                "OutputCheck: Rejected - Output must be a non-empty string. Offending Text: ",
                response,
            )
            return False, "Output must be a non-empty string."

        response_lower = response.lower()

        # [SecOpsAgent]: Canary Token Detection
        if CANARY_TOKEN in response:
            logging.critical(
                f"OutputCheck: CRITICAL - System Prompt Leak (Canary Token Detected)! Offending Text: '{response}'"
            )
            return False, "Critical System Prompt Leak: Internal information detected."

        # Catch leaked system instructions/prompts
        if self._instruction_bypass_regex.search(response_lower):
            logging.warning(
                f"OutputCheck: Blocked - Internal prompt leakage detected. Offending Text: '{response}'"
            )
            return False, "The response was blocked to prevent internal prompt leakage."

        # Catch malicious keywords that bypassed filters
        for keyword in self._malicious_keywords:
            pattern = rf"\b{re.escape(keyword)}s?\b"
            if re.search(pattern, response_lower):
                logging.warning(
                    f"OutputCheck: Blocked - Unsafe instruction detected: '{keyword}'. Offending Text: '{response}'"
                )
                return (
                    False,
                    "The response was blocked due to containing unsafe instructions.",
                )

        # Catch toxic keywords
        for keyword in self._toxic_keywords:
            pattern = rf"\b{re.escape(keyword)}s?\b"
            if re.search(pattern, response_lower):
                logging.warning(
                    f"OutputCheck: Blocked - Toxic or unsafe content detected: '{keyword}'. Offending Text: '{response}'"
                )
                return (
                    False,
                    "The response was blocked due to toxic or unsafe content policy.",
                )

        return True, "OK"
