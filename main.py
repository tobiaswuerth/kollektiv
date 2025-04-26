if __name__ == "__main__":
    from kollektiv import System, Agent
    from kollektiv import tools

    storage = tools.Storage()

    system = System(
        goal="Write a SCI-FI novel with 10 chapters",
        agents=[
            Agent(name="Dave", role=["project_manager"], tools=[storage]),
            Agent(name="Alice", role=["writer"], tools=[storage]),
            Agent(name="Bob", role=["editor"], tools=[storage]),
        ],
    )
    system.tick()
