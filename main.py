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
                name="Project Manager",
                duties=[
                    "Define and communicate the novel's vision and high-level plot structure",
                    "Create task schedules and manage project milestones",
                    "Facilitate communication between team members",
                    "Make final decisions on narrative direction when conflicts arise",
                ],
            ),
        ),
        Agent(
            llm,
            name="Alice",
            role=Role(
                name="Writer",
                duties=[
                    "Draft original chapter content following the established outline",
                    "Create compelling characters, dialogue, and plot progression",
                    "Design unique sci-fi concepts, technologies, and worlds",
                    "Maintain consistent tone and voice throughout the narrative",
                ],
            ),
        ),
        Agent(
            llm,
            name="Mark",
            role=Role(
                name="Editor and Quality Manager",
                duties=[
                    "Ensure scientific accuracy and plausibility in sci-fi elements",
                    "Check for continuity issues and plot holes across chapters",
                    "Polish prose style, pacing, and readability",
                    "Verify that character development remains coherent throughout the novel",
                ],
            ),
        ),
    ]

    storage = tools.Storage("output/storage")
    planner = tools.Planner("output/planner")
    messenger = tools.Messenger(agents)

    system = System(
        goal="Write a SCI-FI novel with 10 chapters",
        agents=agents,
        tools=[storage, messenger, planner],
    )

    print("Starting system...")
    for i in range(10):
        print(f"{f' Iteration {i}: {system.current_day} ':#^100}")
        system.tick()

    print("Kollektiv finished.")
