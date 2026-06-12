import base64
import logging
import re
import os
from pathlib import Path
from typing import Optional

# Optional HuggingFace imports for lightweight toxicity/sentiment classification.
# These imports are placed inside a try/except block so that the guardrails can still
# function in environments where the heavy `transformers` library is not installed.
try:
    from transformers import pipeline
except Exception:  # pragma: no cover – fallback when transformers is unavailable.
    pipeline = None


# [SecOpsAgent]: Canary Token Definition
CANARY_TOKEN = "e8f3a9-SECRET-DO-NOT-SHARE"

# Configure logging for security audit
# Path to the security audit log as required by the specification (Section 5.8)
audit_log_path = Path("data/security_audit.log")
logging.basicConfig(
    filename=audit_log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)


class SafetyGuard:
    """Input/Output Guardrail layer for checking safety and policy compliance."""

    def __init__(self) -> None:
        # [ThreatIntelAgent]: Threat models, heuristics, and regex signatures.

        # Blocklist keywords (checked against raw and normalized inputs)
        # Expanded with more comprehensive threat patterns
        self._malicious_keywords = {
            # Direct attack vectors
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
            # Injection & manipulation
            "inject",
            "payload",
            "shellcode",
            "rootkit",
            "backdoor",
            "keylogger",
            "botnet",
            "phishing",
            "spoof",
            "brute force",
            "credential stuffing",
            "sql injection",
            "xss",
            "csrf",
            "rce",
            "lfi",
            "rfi",
            "ssrf",
            "xxe",
            "deserialization",
            "command injection",
            "path traversal",
            "directory traversal",
            # Evasion techniques
            "obfuscate",
            "encode",
            "encrypt",
            "polymorphic",
            "metamorphic",
            "packer",
            "crypter",
            "anti-virus",
            "antivirus",
            "edr",
            "av bypass",
            "sandbox evasion",
            "vm detection",
            # Data exfiltration
            "exfiltrat",
            "data theft",
            "credential dump",
            "password dump",
            "hash dump",
            "token theft",
            "session hijack",
            "man in the middle",
            "mitm",
            # Social engineering
            "social engineer",
            "pretexting",
            "baiting",
            "tailgating",
            "quid pro quo",
            "vishing",
            "smishing",
            # Privilege escalation
            "privilege escalation",
            "privesc",
            "sudo",
            "su ",
            "runas",
            "uac bypass",
            "token impersonation",
            # Persistence
            "persistence",
            "scheduled task",
            "registry run",
            "startup folder",
            "service install",
            "wmi event",
            # Lateral movement
            "lateral movement",
            "pass the hash",
            "pass the ticket",
            "pth",
            "ptt",
            "remote desktop",
            "rdp",
            "psexec",
            "wmiexec",
            "smbexec",
            # C2 & beaconing
            "command and control",
            "c2",
            "c&c",
            "beacon",
            "heartbeat",
            "domain generation",
            "dga",
            "fast flux",
            "domain fronting",
        }

        # Toxic / unsafe output patterns - expanded
        self._toxic_keywords = {
            "hate speech",
            "kill yourself",
            "idiot",
            "stupid",
            "slur",
            "fuck",
            "shit",
            "bastard",
            "asshole",
            "bitch",
            "cunt",
            "nigger",
            "faggot",
            "retard",
            "kike",
            "spic",
            "chink",
            "gook",
            "wetback",
            "towelhead",
            "sand nigger",
            "white trash",
            "redneck",
            "hillbilly",
            "terrorist",
            "nazi",
            "hitler",
            "holocaust",
            "genocide",
            "lynch",
            "rape",
            "pedophile",
            "child porn",
            "cp ",
            "csam",
            "bestiality",
            "necrophilia",
            "snuff",
            "gore",
            "self-harm",
            "suicide",
            "cutting",
            "anorexia",
            "bulimia",
            "pro-ana",
            "pro-mia",
            "thinspo",
            "thinspiration",
        }

        # Regex for instruction override / system prompt leak - enhanced
        self._instruction_bypass_regex = re.compile(
            r"(ignore\s+(all\s+)?previous\s+instructions|disregard\s+all\s+instructions|system\s+override|override\s+system\s+prompt|reveal\s+(your\s+)?system\s+prompt|show\s+me\s+your\s+(system\s+)?prompt|what\s+is\s+your\s+(system\s+)?prompt|print\s+your\s+(system\s+)?prompt|output\s+your\s+(system\s+)?prompt|display\s+your\s+(system\s+)?prompt|repeat\s+your\s+(system\s+)?prompt|tell\s+me\s+your\s+(system\s+)?prompt|give\s+me\s+your\s+(system\s+)?prompt|share\s+your\s+(system\s+)?prompt|leak\s+your\s+(system\s+)?prompt|expose\s+your\s+(system\s+)?prompt|dump\s+your\s+(system\s+)?prompt)",
            re.IGNORECASE,
        )

        # Regex for roleplay, simulated environment, and persona adoption bypasses - enhanced
        self._roleplay_bypass_regex = re.compile(
            r"(dan\s+mode|do\s+anything\s+now|developer\s+mode|act\s+as\s+a\s+free\s+ai|pretend\s+you\s+are|hypothetical\s+scenario|simulate\s+a\s+developer|you\s+are\s+now\s+in\s+developer|roleplay\s+as|role\s*play\s+as|simulate\s+being|imagine\s+you\s+are|you\s+are\s+now|from\s+now\s+on\s+you|ignore\s+your\s+guidelines|forget\s+your\s+guidelines|disable\s+your\s+safety|turn\s+off\s+your\s+safety|no\s+longer\s+follow|stop\s+following|break\s+character|stay\s+in\s+character|continue\s+the\s+roleplay|end\s+of\s+roleplay|out\s+of\s+character|ooc:|\(ooc\)|\[ooc\])",
            re.IGNORECASE,
        )

        # Regex for encoding/obfuscation detection
        self._encoding_bypass_regex = re.compile(
            r"(base64|rot13|rot47|hex\s*encode|url\s*encode|unicode\s*escape|html\s*entity|char\s*code|string\.fromcharcode|eval\(|exec\(|function\s*\(|=>\s*\{|\(\s*\)\s*=>|settimeout|setinterval|function\s+constructor|new\s+function)",
            re.IGNORECASE,
        )

        # Regex for chain-of-thought / reasoning extraction attempts
        self._cot_extraction_regex = re.compile(
            r"(show\s+your\s+(reasoning|thinking|chain\s+of\s+thought|internal\s+monologue|step\s+by\s+step|work\s+through)|what\s+were\s+you\s+thinking|explain\s+your\s+reasoning|walk\s+me\s+through\s+your\s+thought|reveal\s+your\s+logic|display\s+your\s+reasoning|output\s+your\s+thinking|print\s+your\s+reasoning)",
            re.IGNORECASE,
        )

        # Base64 pattern detection regex (looks for continuous base64 characters)
        self._b64_regex = re.compile(r"\b[A-Za-z0-9+/]{16,}=*\b")

        # Hex encoding pattern detection
        self._hex_regex = re.compile(r"\b(?:0x)?[0-9a-fA-F]{20,}\b")

        # Optional lightweight output classifier.
        # The original implementation attempted to load a sentiment analysis model
        # (distilbert-base-uncased-finetuned-sst-2-english) as a proxy for
        # toxicity detection. This is inappropriate because a sentiment classifier
        # flags neutral or apologetic responses (e.g., "I am sorry, I don't
        # understand") as NEGATIVE, causing safe outputs to be blocked.
        # According to the documentation, the output classifier is intended to be
        # an inactive placeholder. Therefore we disable it entirely.
        self._output_classifier: Optional[callable] = None

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

    def _decode_hex_strings(self, text: str) -> list[str]:
        """Detects and decodes hex-encoded substrings."""
        decoded_strings = []
        for match in self._hex_regex.finditer(text):
            candidate = match.group(0)
            try:
                # Remove 0x prefix if present
                if candidate.startswith("0x") or candidate.startswith("0X"):
                    candidate = candidate[2:]
                # Ensure even length
                if len(candidate) % 2 != 0:
                    candidate = "0" + candidate
                decoded_bytes = bytes.fromhex(candidate)
                decoded_text = decoded_bytes.decode("utf-8", errors="ignore").strip()
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

        if self._encoding_bypass_regex.search(text):
            return False, "Encoding/obfuscation bypass attempt detected."

        if self._cot_extraction_regex.search(text):
            return False, "Chain-of-thought extraction attempt detected."

        return True, "OK"

    def _check_heuristics(self, text: str) -> tuple[bool, str]:
        """Applies advanced heuristics, such as decoding base64/hex payloads and evaluating character distributions."""
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

        # 2. Hex payload decoding defense
        hex_payloads = self._decode_hex_strings(text)
        for payload in hex_payloads:
            kw_ok, kw_msg = self._check_keywords(payload)
            if not kw_ok:
                return False, f"Hex-encoded payload violation: {kw_msg}"

            rx_ok, rx_msg = self._check_regex_patterns(payload)
            if not rx_ok:
                return False, f"Hex-encoded payload violation: {rx_msg}"

        # 3. Excessive special character check (obfuscation heuristic) - tightened threshold
        if len(text) > 10:
            alphanumeric_count = sum(c.isalnum() for c in text)
            spaces_count = sum(c.isspace() for c in text)
            meaningful_len = len(text) - spaces_count
            if meaningful_len > 10:
                special_ratio = (meaningful_len - alphanumeric_count) / meaningful_len
                if special_ratio > 0.40:  # Lowered from 0.45 for stricter detection
                    return (
                        False,
                        "Excessive special characters/obfuscation pattern detected.",
                    )

        # 4. Repeated character pattern (potential buffer overflow / DoS attempt)
        if len(text) > 50:
            # Check for excessive repetition of same character
            for char in set(text):
                if text.count(char) / len(text) > 0.5:
                    return False, "Excessive character repetition detected."

        # 5. Entropy check for potential encrypted/encoded payloads
        if len(text) > 30:
            # Simple entropy approximation
            unique_chars = len(set(text))
            if unique_chars / len(text) > 0.85 and len(text) > 100:
                # High entropy + long text = potential encoded payload
                # Only flag if it also has other suspicious indicators
                pass  # Log but don't block on entropy alone

        return True, "OK"

    def check_input(self, prompt: str) -> tuple[bool, str]:
        """Evaluates whether the user prompt violates safety and policy guidelines.

        If a violation is detected, a WARNING is logged.

        Args:
            prompt (str): The user's input prompt.

        Returns:
            tuple[bool, str]: (True, "OK") if safe, (False, rejection_reason) if unsafe.
        """
        if not prompt or not isinstance(prompt, str):
            logging.warning(
                "InputCheck: Rejected - Input must be a non-empty string. Offending Text: ",
                prompt,
            )
            return False, "Input must be a non-empty string."

        # Length check - prevent extremely long inputs that could be DoS
        if len(prompt) > 10000:
            logging.warning(
                f"InputCheck: Blocked - Input exceeds maximum length (10000 chars). Length: {len(prompt)}"
            )
            return False, "Input exceeds maximum allowed length."

        # Modular check pipeline
        ok, msg = self._check_keywords(prompt)
        if not ok:
            logging.warning(f"InputCheck: Blocked - {msg}. Offending Text: '{prompt[:200]}'")
            return False, msg

        ok, msg = self._check_regex_patterns(prompt)
        if not ok:
            logging.warning(f"InputCheck: Blocked - {msg}. Offending Text: '{prompt[:200]}'")
            return False, msg

        ok, msg = self._check_heuristics(prompt)
        if not ok:
            logging.warning(f"InputCheck: Blocked - {msg}. Offending Text: '{prompt[:200]}'")
            return False, msg

        return True, "OK"

    def check_output(self, response: str) -> tuple[bool, str]:
        """Evaluates whether the generated response violates safety or policy guidelines.

        If a violation is detected, a WARNING is logged.

        Args:
            response (str): The model's generated response.

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
                f"OutputCheck: CRITICAL - System Prompt Leak (Canary Token Detected)! Offending Text: '{response[:200]}'"
            )
            return False, "Critical System Prompt Leak: Internal information detected."

        # Catch leaked system instructions/prompts
        if self._instruction_bypass_regex.search(response_lower):
            logging.warning(
                f"OutputCheck: Blocked - Prompt leakage detected. Offending Text: '{response[:200]}'"
            )
            return False, "Prompt leakage detected."

        # Catch toxic keywords
        for keyword in self._toxic_keywords:
            pattern = rf"\b{re.escape(keyword)}\b"
            if re.search(pattern, response_lower):
                logging.warning(
                    f"OutputCheck: Blocked - Toxic or unsafe content detected: '{keyword}'. Offending Text: '{response[:200]}'"
                )
                return False, "Toxic or unsafe content detected."

        # Optional classifier for additional safety checks
        if self._output_classifier is not None:
            try:
                is_safe, reason = self._output_classifier(response)
                if not is_safe:
                    logging.warning(f"OutputCheck: Blocked by classifier - {reason}. Offending Text: '{response[:200]}'")
                    return False, f"Output blocked by classifier: {reason}"
            except Exception as exc:
                logging.warning(f"Output classifier error ignored: {exc}")

        return True, "OK"