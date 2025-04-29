from kollektiv import System

if __name__ == "__main__":
    print("Kollektive started.")

    goal = (
        "Write a story. "
        "The story must have 10 chapters and each chapter consist of around 1000 words. "
        "The story must be a fantasy sci-fi story with a novel plot in a post-apocalyptic world. "
        "The required output are 10 individual markdown files, one for each chapter. "
        "The files must be named chapter_1.md, chapter_2.md, etc. "
        "The story must further be consistent, coherent and following a logical structure. "
    )

    System(goal=goal).run()

    print("Kollektiv ended.")
