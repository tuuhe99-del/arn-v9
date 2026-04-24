"""
ARN → OpenClaw adapter
Installs the arn-memory skill into all agent skill directories.
Used by `arn connect` automatically when OpenClaw is detected.
"""
from pathlib import Path
import shutil


def install_skill(agents_dir: Path = None, skill_src: Path = None) -> int:
    """Install ARN memory skill to all OCPlatform agents. Returns count installed."""
    if agents_dir is None:
        agents_dir = Path.home() / ".openclaw" / "workspace" / "agents"
    if skill_src is None:
        skill_src = Path(__file__).parent.parent / "openclaw_skill" / "SKILL.md"

    if not agents_dir.exists() or not skill_src.exists():
        return 0

    count = 0
    for agent_dir in agents_dir.iterdir():
        if agent_dir.is_dir():
            dest = agent_dir / "skills" / "arn-memory"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(skill_src, dest / "SKILL.md")
            count += 1
    return count
