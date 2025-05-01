from typing import Optional, Self
from pydantic import BaseModel, Field
from copy import deepcopy


def find_node_recursive(node: "Node", target_node_id: int) -> Optional["Node"]:
    if node.id == target_node_id:
        return node
    for child in node.children:
        found = find_node_recursive(child, target_node_id)
        if found:
            return found
    return None


class Node:
    def __init__(
        self,
        name: str,
        description: str,
    ):
        self.id: int = id(self)
        self.name: str = name
        self.description: str = description
        self.parent: Optional["Node"] = None
        self.children: list["Node"] = []

    @property
    def parents(self) -> list["Node"]:
        if self.parent is None:
            return []
        return [self.parent] + self.parent.parents

    @property
    def root(self) -> "Node":
        if self.parent is None:
            return self
        return self.parent.root

    def add_child(self, child: "Node") -> Self:
        self.children.append(child)
        child.parent = self
        return self

    def add_children(self, children: list["Node"]) -> Self:
        for child in children:
            self.add_child(child)
        return self

    def remove_child(self, child: "Node"):
        self.children.remove(child)
        child.parent = None

    def find_node(self, node_id: int) -> Optional["Node"]:
        return find_node_recursive(self, node_id)

    def to_json(self, include_parents=True, include_children=True) -> dict:
        parent = (
            None
            if self.parent is None
            else (
                "..."
                if not include_parents
                else self.parent.to_json(include_parents=True, include_children=False)
            )
        )
        children = (
            []
            if len(self.children) == 0
            else (
                "..."
                if not include_children
                else [child.to_json(False, include_children) for child in self.children]
            )
        )

        return {
            "parent": parent,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "children": children,
        }

    def clone(self) -> "Node":
        return deepcopy(self)

    def __str__(self):
        return f"Node({self.id}, {self.name}, children=[{len(self.children)}x...])"


class NodeModel(BaseModel):
    name: str = Field(description="Name of the node")
    description: str = Field(description="Description of the node")

    def to_node(self) -> Node:
        return Node(
            name=self.name,
            description=self.description,
        )


class NodeListModel(BaseModel):
    nodes: list[NodeModel] = Field(description="List of nodes")

    def to_nodes(self) -> list[Node]:
        return [node.to_node() for node in self.nodes]
