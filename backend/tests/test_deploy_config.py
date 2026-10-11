"""render.yaml stays in sync with Settings and backend/.env.example, and holds no secrets."""
import yaml

from app.core.config import BACKEND_DIR, REPO_ROOT, Settings

SECRETS = {"DATABASE_URL", "GOOGLE_API_KEY", "GROQ_API_KEY"}
GENERATED = {"SALES_API_KEY"}
# Settings fields that are not set on Render, and why
LEFT_OUT = {
    "TEST_DATABASE_URL": "tests only",
    "KB_PATH": "ingest only; the PDF is not in the image",
    "LANGSMITH_API_KEY": "tracing is off; set it in the Dashboard if you turn it on",
    "LANGSMITH_HIDE_INPUTS": "tracing is off on Render; T13 decides the privacy settings",
    "LANGSMITH_HIDE_OUTPUTS": "tracing is off on Render; T13 decides the privacy settings",
    "LANGSMITH_HIDE_METADATA": "tracing is off on Render; T13 decides the privacy settings",
}
# Plain values meant to differ from backend/.env.example in production, and why (T13: CORS_ORIGINS)
PROD_DIFFERS: dict[str, str] = {}


def service() -> dict:
    blueprint = yaml.safe_load((REPO_ROOT / "render.yaml").read_text(encoding="utf-8"))
    [web] = blueprint["services"]
    return web


def env_vars() -> dict[str, dict]:
    return {item["key"]: item for item in service()["envVars"]}


def env_example() -> dict[str, str]:
    """KEY=value lines of backend/.env.example (comments and blank lines skipped)."""
    values = {}
    for line in (BACKEND_DIR / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def test_free_docker_service_in_singapore():
    web = service()
    assert (web["type"], web["runtime"], web["plan"]) == ("web", "docker", "free")
    assert web["region"] == "singapore"  # closest to the Neon project; cannot be changed later
    assert web["healthCheckPath"] == "/api/health/live"
    assert (web["dockerfilePath"], web["dockerContext"]) == ("./Dockerfile", ".")
    assert web["autoDeployTrigger"] == "commit"


def test_every_setting_is_on_render_or_left_out_on_purpose():
    settings_keys = {name.upper() for name in Settings.model_fields}
    assert settings_keys == set(env_vars()) | set(LEFT_OUT)
    assert not set(env_vars()) & set(LEFT_OUT)


def test_secrets_have_no_value_in_the_file():
    variables = env_vars()
    for key in SECRETS:
        assert variables[key] == {"key": key, "sync": False}
    for key in GENERATED:
        assert variables[key] == {"key": key, "generateValue": True}
    plain = set(variables) - SECRETS - GENERATED
    assert all("value" in variables[key] for key in plain)


def test_render_keys_are_documented_in_env_example():
    assert set(env_vars()) <= set(env_example())


def test_plain_values_match_env_example():
    example = env_example()
    for key, item in env_vars().items():
        if "value" not in item or key in PROD_DIFFERS or not example.get(key):
            continue
        assert str(item["value"]) == example[key], f"{key} differs between render.yaml and .env.example"
