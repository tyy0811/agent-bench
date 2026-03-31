"""Tests for security configuration models."""

import pytest
from pydantic import ValidationError

from agent_bench.core.config import AppConfig


class TestSecurityConfig:
    def test_security_config_has_defaults(self):
        """SecurityConfig is present on AppConfig with sane defaults."""
        config = AppConfig()
        assert config.security.injection.enabled is True
        assert config.security.injection.action == "block"
        assert config.security.injection.tiers == ["heuristic", "classifier"]
        assert config.security.pii.enabled is True
        assert config.security.pii.mode == "redact"
        assert "EMAIL" in config.security.pii.redact_patterns
        assert config.security.pii.use_ner is False
        assert config.security.output.enabled is True
        assert config.security.output.pii_check is True
        assert config.security.output.url_check is True
        assert config.security.output.blocklist == []
        assert config.security.audit.enabled is True
        assert config.security.audit.path == "logs/audit.jsonl"

    def test_security_config_from_yaml(self, tmp_path):
        """Security config loads from YAML correctly."""
        import yaml
        config_data = {
            "security": {
                "injection": {"enabled": False, "action": "warn"},
                "pii": {"mode": "passthrough", "use_ner": True},
                "audit": {"path": "custom/audit.jsonl", "max_size_mb": 50},
            }
        }
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        from agent_bench.core.config import load_config
        config = load_config(path=yaml_path)
        assert config.security.injection.enabled is False
        assert config.security.injection.action == "warn"
        assert config.security.pii.mode == "passthrough"
        assert config.security.pii.use_ner is True
        assert config.security.audit.path == "custom/audit.jsonl"
        assert config.security.audit.max_size_mb == 50

    def test_injection_action_values(self):
        """Injection action accepts block, warn, flag."""
        from agent_bench.core.config import InjectionConfig
        for action in ("block", "warn", "flag"):
            cfg = InjectionConfig(action=action)
            assert cfg.action == action

    def test_pii_mode_values(self):
        """PII mode accepts redact, detect_only, passthrough."""
        from agent_bench.core.config import PIIConfig
        for mode in ("redact", "detect_only", "passthrough"):
            cfg = PIIConfig(mode=mode)
            assert cfg.mode == mode

    def test_injection_action_rejects_invalid(self):
        """Invalid injection action raises ValidationError."""
        from agent_bench.core.config import InjectionConfig
        with pytest.raises(ValidationError):
            InjectionConfig(action="drop")

    def test_pii_mode_rejects_invalid(self):
        """Invalid PII mode raises ValidationError."""
        from agent_bench.core.config import PIIConfig
        with pytest.raises(ValidationError):
            PIIConfig(mode="whatever")

    def test_invalid_action_in_yaml_rejected(self, tmp_path):
        """A YAML typo in injection.action must not silently pass."""
        import yaml
        config_data = {"security": {"injection": {"action": "yolo"}}}
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(yaml.dump(config_data))

        from agent_bench.core.config import load_config
        with pytest.raises(ValidationError):
            load_config(path=yaml_path)
