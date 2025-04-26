if __name__ == "__main__":
    print("Starting Kollektiv...")

    from kollektiv import System, Agent
    from kollektiv import tools

    agents = [
        Agent(name="Dave", role=["project_manager"]),
        Agent(name="Alice", role=["writer"]),
        Agent(name="Bob", role=["editor"]),
    ]

    storage = tools.Storage()
    messenger = tools.Messenger(agents)
    for a in agents:
        a.tools = [storage, messenger]

    system = System(
        goal="Write a SCI-FI novel with 10 chapters",
        agents=agents,
    )

    print("Starting system...")
    for _ in range(5):
        system.tick()

    print("Kollektiv finished.")
