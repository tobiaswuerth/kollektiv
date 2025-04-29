from typing import Optional
from pydantic import BaseModel, Field


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
        order: int,
    ):
        self.id: int = id(self)
        self.name: str = name
        self.description: str = description
        self.order: int = order
        self.parent: Optional["Node"] = None
        self.children: list["Node"] = []

    def add_child(self, child: "Node"):
        self.children.append(child)
        child.parent = self

    def add_children(self, children: list["Node"]):
        for child in children:
            self.add_child(child)

    def remove_child(self, child: "Node"):
        self.children.remove(child)
        child.parent = None

    def find_node(self, node_id: int) -> Optional["Node"]:
        return find_node_recursive(self, node_id)

    def siblings(self) -> list["Node"]:
        if self.parent is None:
            return []
        return [child for child in self.parent.children if child != self]

    def parents(self) -> list["Node"]:
        if self.parent is None:
            return []
        return [self.parent] + self.parent.parents()

    def to_json(self, include_children=True, include_parents=True) -> dict:
        parent = (
            None
            if self.parent is None
            else (
                "..."
                if not include_parents
                else self.parent.to_json(include_children=False, include_parents=True)
            )
        )
        children = (
            "..."
            if not include_children
            else [
                child.to_json(include_children, False)
                for child in self.children
            ]
        )

        return {
            "parent": parent,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.order,
            "children": children,
        }

    def __str__(self):
        return f"Node({self.id}, ##{self.order}, {self.name}, children=[{len(self.children)}x...])"


class NodeModel(BaseModel):
    name: str = Field(description="Name of the node")
    description: str = Field(description="Description of the node")
    order: int = Field(description="Priority of the node, lower come first")

    def to_node(self) -> Node:
        return Node(
            name=self.name,
            description=self.description,
            order=self.order,
        )


class NodeListModel(BaseModel):
    nodes: list[NodeModel] = Field(description="List of nodes")

    def to_nodes(self) -> list[Node]:
        return [node.to_node() for node in self.nodes]
