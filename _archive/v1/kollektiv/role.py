class Role:
    def __init__(self, name: str, duties: list[str]):
        self.name:str = name
        self.duties:list[str] = duties
        
    def __str__(self):
        return f"Role({self.name}, duties={self.duties})"