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
                    "Evaluate completed chapters against quality standards",
                    "Determine when each chapter meets requirements for completion",
                    "Judge if story elements align with the overall vision",
                    "Make final decisions on whether to proceed or request revisions",
                    "Track overall novel progress and completion status"
                ],
            ),
        ),
        Agent(
            llm,
            name="Alice",
            role=Role(
                name="Writer",
                duties=[
                    "Draft original chapter content following your idea of an outline",
                    "Create compelling characters, dialogue, and plot progression",
                    "Edit and revise content based on feedback",
                    "Maintain consistent tone and voice throughout the narrative",
                    "Polish prose style and improve readability"
                ],
            ),
        ),
    ]

    storage = tools.Storage("output/storage")
    planner = tools.Planner("output/planner")
    messenger = tools.Messenger(agents)

    system = System(
        goal=\
"""Write a SCI-FI novel with 10 chapters containing a 1000 words each.
The final output are 10 markdown files in the output directory following the naming scheme: chapter_1.md, chapter_2.md, ... chapter_10.md.""",
        agents=agents,
        tools=[storage, messenger, planner],
    )

    print("Starting system...")
    for i in range(300):
        print(f"{f' Iteration {i}: {system.current_day} ':#^100}")
        system.tick()

    print("Kollektiv finished.")
