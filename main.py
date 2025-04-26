if __name__ == "__main__":
    print("Starting Kollektiv...")

    from kollektiv import System, Agent, Role
    from kollektiv import LLMClient, tools

    llm = LLMClient()

    agents = [
        Agent(
            llm,
            name="Dave",
            role=Role(
                "Project Manager",
                [
                    "Manage the project",
                    "Give tasks to other agents",
                    "Keep track of the project progress",
                ],
            ),
        ),
        Agent(
            llm,
            name="Simone",
            role=Role(
                "Writer",
                [
                    "Create a plot outline",
                    "Write the chapters",
                    "Write the novel to file",
                ],
            ),
        ),
        Agent(
            llm,
            name="Paul",
            role=Role(
                "Editor",
                [
                    "Edit the novel",
                    "Provide feedback to the writer",
                    "Ensure the novel is coherent and engaging",
                ],
            ),
        ),
        Agent(
            llm,
            name="Tiara",
            role=Role(
                "Researcher",
                [
                    "Research the SCI-FI genre",
                    "Provide references and inspiration for the novel",
                    "Ensure the novel is scientifically accurate",
                ],
            ),
        ),
    ]

    storage = tools.Storage("output")
    messenger = tools.Messenger(agents)
    all_tools = [storage, messenger]
    for a in agents:
        a.tools = all_tools

    system = System(
        goal="Write a SCI-FI novel with 10 chapters",
        agents=agents,
        tools=all_tools,
    )

    print("Starting system...")
    for _ in range(10):
        print(f"{f' Iteration {system.time} ':#^100}")
        system.tick()

    print("Kollektiv finished.")
