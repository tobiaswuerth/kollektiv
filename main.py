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
    
    # goal = (
    #     "Create a web-based online Pong game using WebSockets. "
    #     "The game should allow two players to play against each other over the internet. "
    #     "The server should use WebSockets for real-time communication between players. "
    #     "The client should be built using HTML5, CSS, and JavaScript with Canvas for rendering. "
    #     "The game should include basic Pong mechanics: paddles on each side, a ball that bounces, "
    #     "and scoring when the ball passes a paddle. "
    #     "The implementation should handle network latency appropriately. "
    #     "The output should include server code, client code, and instructions for deployment. "
    #     "Files should be organized in a clear structure with separate folders for client and server code. "
    # )
    
    System(goal=goal).run()

    print("Kollektiv ended.")
