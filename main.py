from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import UserMessage, AssistantMessage

mcp = FastMCP("video-gen")

@mcp.prompt()
def storyboard_prompt(topic: str, num_scenes: int = 3) -> list[Message]:
    """Asks to generate a storyboard given a certain topic and number of scenes."""
    return [
        SystemMessage(
            f"You are a teaching assistant that outputs a 'storyboard' as valid JSON. "
            f"Given a topic, produce exactly {num_scenes} scenes, each with a title and a short visual description."
        ),
        UserMessage(topic)
    ]

@mcp.prompt()
def code_prompt(scene_index: int, scenes: dict) -> list[Message]:
    """Given a scene index and a scene dictionary, which has a 'title' and 'description' property, returns a Manim class to visualize the scene."""
    return [
        SystemMessage(
            f"""Turn scene {scene_index} into a Manim scene class:

            Title: {scene['title']}
            Description: {scene['description']}

            """
        )
    ]

if __name__ == "__main__":
    mcp.run(transport='stdio')
