from arn.plugin import ARNPlugin


def test_identity_rules_procedure_error_context_packet(tmp_path):
    p = ARNPlugin(agent_id="developer", data_root=str(tmp_path), embedding_tier="nano", auto_consolidate=False)
    try:
        p.set_identity("developer", name="Koda", role="coding and tests", must=["run tests"], must_not=["claim success without proof"])
        p.add_rule("developer", "Do not claim success unless tests pass", priority="critical")
        p.add_procedure("release-audit", ["compile python", "run pytest"], success="tests pass", agent="developer")
        p.add_error_lesson("CLI handler was missing", fix="add handler", lesson="smoke test documented commands", agent="developer")

        packet = p.build_context_packet(query="audit a release", agent="developer", task="audit ARN release", max_tokens=1000)
        assert "## Identity" in packet
        assert "Koda" in packet
        assert "## Rules" in packet
        assert "Do not claim success" in packet
        assert "## Current Task" in packet
        assert "audit ARN release" in packet
        assert "## Relevant Procedures" in packet
        assert "release-audit" in packet
        assert "## Past Errors / Lessons" in packet
        assert "CLI handler" in packet
    finally:
        p.shutdown()


def test_typed_store_metadata(tmp_path):
    p = ARNPlugin(agent_id="default", data_root=str(tmp_path), embedding_tier="nano", auto_consolidate=False)
    try:
        p.store("Mohamed prefers Python", memory_type="preference", scope="user", priority="high")
        rows = p._recent_by_type("preference", scopes=["user"], limit=5)
        assert len(rows) == 1
        assert rows[0]["scope"] == "user"
        assert rows[0]["priority"] == "high"
        assert "Python" in rows[0]["content"]
    finally:
        p.shutdown()


def test_cross_agent_share_inbox_and_context_packet(tmp_path):
    dev = ARNPlugin(agent_id="developer", data_root=str(tmp_path), embedding_tier="nano", auto_consolidate=False)
    try:
        result = dev.share_memory(
            "CLI smoke tests passed after the command handler fix.",
            to_agents=["manager", "researcher"],
            from_agent="developer",
            task="ARN release",
        )
        assert result["ok"] is True
        assert {d["agent"] for d in result["deliveries"]} >= {"developer", "manager", "researcher"}
    finally:
        dev.shutdown()

    manager = ARNPlugin(agent_id="manager", data_root=str(tmp_path), embedding_tier="nano", auto_consolidate=False)
    try:
        inbox = manager.list_shared_memories(agent="manager", direction="inbox", task="ARN release")
        assert len(inbox) == 1
        assert "CLI smoke tests passed" in inbox[0]["content"]
        packet = manager.build_context_packet(query="what happened with tests?", agent="manager", task="ARN release", max_tokens=1000)
        assert "## Shared Agent Notes" in packet
        assert "developer" in packet
        assert "CLI smoke tests passed" in packet
    finally:
        manager.shutdown()
