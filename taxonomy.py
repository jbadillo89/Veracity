import json
from pathlib import Path
KNOWLEDGE_DIRECTORY = Path(__file__).resolve().parent.parent / "knowledge"
def load_knowledge(filename: str) -> dict:
    with (KNOWLEDGE_DIRECTORY / filename).open(encoding="utf-8") as source:
        return json.load(source)
class Taxonomy:
    def __init__(self) -> None:
        data = load_knowledge("categories.json")
        self.categories = {item["category"]: item for item in data["categories"]}
        self.separation_contexts = tuple(data["separation_contexts"])
        codes = [item["code"] for item in data["categories"]]
        if len(self.categories) != len(data["categories"]) or len(set(codes)) != len(codes):
            raise ValueError("Duplicate taxonomy category or code.")
        for item in self.categories.values():
            if len(set(item["subcategories"])) != len(item["subcategories"]):
                raise ValueError("Duplicate taxonomy subcategory.")
        if "UNCLASSIFIED" not in self.categories:
            raise ValueError("The taxonomy requires UNCLASSIFIED.")
    def validate(self, category: str, subcategory: str | None) -> bool:
        if category not in self.categories:
            return False
        allowed = self.categories[category]["subcategories"]
        return subcategory in allowed if allowed else subcategory is None
    def require_valid(self, category: str, subcategory: str | None) -> None:
        if not self.validate(category, subcategory):
            raise ValueError(f"Invalid taxonomy pair: {category}/{subcategory}")
