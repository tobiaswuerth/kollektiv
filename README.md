# Kollektiv

> Inspired by [generative_agents](https://github.com/joonspk-research/generative_agents)

I'd like to experiment with AI agents.

I see a lot of interesting aspects in a project like this, ranging from interactions with LLMs, to RAG, system lifecycles, virtual environments and much more.

The goal is:
- have a team of agents
- each agent has a persona with certain characteristics, skills and access to tools
- the agents interact with each other, trying to solve a common problem

---

# Experiments

in the `_archive` folder I currently have some of the older experiments.
v1 is a simple multi-agent system that tries to achieve some kind of goal. .. it didn't perform well.
v2 is a hello-world example basically of langchain/langgraph, but this became cumbersome when I wanted to generate dynamic graphs.

In the end I figured I'm better of just writing my own code where I have full control. 
I built a v3 (which is now on the main branch) that tries to improve on things.

I noticed that one of the key challenges is to break a problem down into sub-steps.
I explicitly focus the recent efforts on building a system that is able to do just that.

Here are some of the results.

For example, given this as a goal:
```python
    goal = (
        "Develop an online Pong game with two-player support over the internet. "
        "Include basic mechanics: paddles, bouncing ball, and scoring. "
        "Provide client and server files."
    )
```

It produced this plan:

![project_plan_with_tasks](https://github.com/user-attachments/assets/4e0212c6-5882-4a1b-b8bd-a30b517fdf09)

Another example:
```python
    goal = (
        "Write a story. "
        "The story must have 10 chapters and each chapter consist of around 1000 words. "
        "The story must be a fantasy sci-fi story with a novel plot in a post-apocalyptic world. "
        "The required output are 10 individual markdown files, one for each chapter. "
        "The files must be named chapter_1.md, chapter_2.md, etc. "
        "Each file only contains the chapter title and the content of the chapter. "
        "The story must further be consistent, coherent and following a logical structure. "
    )
```

It produced this plan:

![project_plan_with_tasks](https://github.com/user-attachments/assets/91d1f1d1-42cc-47b3-8a76-d41ef05f7b17)

I think this is a good start.
It doesn't help to have multiple agents if none of them are great at planing a project.
My hope is to built upon this, to get to a point where one or multiple agents are able to achieve the goal.

---

# Setup
on Windows

```bash
py -m venv .venv
.\.venv\Scripts\activate
py -m pip install --upgrade pip
pip3 install -r .\requirements.txt
```

# Run

```bash
py .\main.py
```
