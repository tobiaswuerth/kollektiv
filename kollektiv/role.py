class Role:
    def __init__(self, name: str, objectives: list[str]):
        self.name:str = name
        self.objectives:list[str] = objectives
        
    def __str__(self):
        return f"Role(name={self.name}, objectives={self.objectives})"