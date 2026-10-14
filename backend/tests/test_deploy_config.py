"""render.yaml stays in sync with Settings and backend/.env.example, and holds no secrets. The keep-warm workflow
pings the live API on its schedule (T14)."""
import yaml

from app.core.config import BACKEND_DIR, REPO_ROOT, Settings

KEEP_WARM = REPO_ROOT / ".github" / "workflows" / "keep-warm.yml"
IST_OFFSET_MIN = 5 * 60 + 30  # GitHub reads cron in UTC; IST is UTC + 5:30

SECRETS = {"DATABASE_URL", "GOOGLE_API_KEY", "GROQ_API_KEY"}
GENERATED = {"SALES_API_KEY"}
# Settings fields that are not set on Render, and why
LEFT_OUT = {
    "TEST_DATABASE_URL": "tests only",
    "KB_PATH": "ingest only; the PDF is not in the image",
    "LANGSMITH_API_KEY": "tracing is off; set it in the Dashboard if you turn it on",
    "LANGSMITH_HIDE_INPUTS": "tracing stays off on Render (T13b); set it to true before turning tracing on",
    "LANGSMITH_HIDE_OUTPUTS": "tracing stays off on Render (T13b); set it to true before turning tracing on",
    "LANGSMITH_HIDE_METADATA": "tracing stays off on Render (T13b); set it to true before turning tracing on",
}
# Plain values meant to differ from backend/.env.example in production, and why
PROD_DIFFERS: dict[str, str] = {
    "CLIENT_IP_HEADER": "Cloudflare sets CF-Connecting-IP on Render; local requests come through no proxy",
    "CORS_ORIGINS": "the Vercel production URL; local runs use http://localhost:5173",
}


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


def test_cors_allows_only_the_vercel_production_site():
    # T13b: read with the backend's own parser, so a trailing slash, a "*", a preview URL or localhost fails here
    value = env_vars()["CORS_ORIGINS"]["value"]
    assert Settings(cors_origins=value).cors_origin_list == ["https://vantis-sales-agent.vercel.app"]


def test_tracing_stays_off_on_render():
    # T13b: no chat text goes to LangSmith from the live site
    variables = env_vars()
    assert variables["LANGSMITH_TRACING"]["value"] == "false"
    assert "LANGSMITH_API_KEY" not in variables


def cron_values(field: str, low: int, high: int) -> list[int]:
    """The values one cron field matches. Handles "*", "n" and "a-b", each with an optional "/step"."""
    span, _, step = field.partition("/")
    if span == "*":
        first, last = low, high
    elif "-" in span:
        first, last = map(int, span.split("-"))
    else:
        first = last = int(span)
    return list(range(first, last + 1, int(step) if step else 1))


def test_keep_warm_pings_the_light_check_every_10_minutes_from_8_to_23_ist():
    # T14: keeps Render awake for the reviewers; the light check, so Neon still sleeps between chats
    workflow = yaml.safe_load(KEEP_WARM.read_text(encoding="utf-8"))
    triggers = workflow.get(True, workflow.get("on"))  # PyYAML reads the key "on" as the boolean True
    minutes_ist = set()
    for item in triggers["schedule"]:
        minute, hour, day, month, weekday = item["cron"].split()
        assert (day, month, weekday) == ("*", "*", "*")
        for h in cron_values(hour, 0, 23):
            for m in cron_values(minute, 0, 59):
                minutes_ist.add((h * 60 + m + IST_OFFSET_MIN) % (24 * 60))
    assert sorted(minutes_ist) == list(range(8 * 60, 23 * 60 + 1, 10))  # 08:00, 08:10, …, 23:00
    assert "workflow_dispatch" in triggers
    assert workflow["permissions"] == {}
    [job] = workflow["jobs"].values()
    [step] = job["steps"]
    assert '"$API_URL/api/health/live"' in step["run"]
    assert step["env"]["API_URL"].startswith("https://") and not step["env"]["API_URL"].endswith("/")
