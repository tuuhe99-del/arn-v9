from arn import ARNPlugin

with ARNPlugin(agent_id="demo") as memory:
    memory.store("Mohamed prefers Python", importance=0.9, tags=["preference", "coding"])
    results = memory.recall("what does the user like to code in?", top_k=3)
    for hit in results:
        print(hit["score"], hit["content"])
