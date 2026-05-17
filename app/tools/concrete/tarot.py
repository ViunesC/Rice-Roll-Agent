import random
from typing import Any

from tools.tool import Tool, ToolParameter
from tools.registry import ToolRegistry


MAJOR_ARCANA = [
    {
        "name": "The Fool",
        "meaning": "new beginnings, spontaneity, innocence, and taking a leap of faith",
        "reversed_meaning": "recklessness, hesitation, poor judgment, or an avoidable risk",
    },
    {
        "name": "The Magician",
        "meaning": "willpower, skill, manifestation, and focused action",
        "reversed_meaning": "blocked potential, manipulation, scattered energy, or unclear intent",
    },
    {
        "name": "The High Priestess",
        "meaning": "intuition, mystery, inner knowledge, and quiet perception",
        "reversed_meaning": "secrets, confusion, ignored intuition, or hidden motives",
    },
    {
        "name": "The Empress",
        "meaning": "abundance, creativity, care, beauty, and growth",
        "reversed_meaning": "creative block, dependence, neglect, or overgiving",
    },
    {
        "name": "The Emperor",
        "meaning": "structure, authority, stability, discipline, and protection",
        "reversed_meaning": "rigidity, control issues, instability, or misuse of power",
    },
    {
        "name": "The Hierophant",
        "meaning": "tradition, teaching, shared values, and spiritual guidance",
        "reversed_meaning": "rebellion, stale convention, private belief, or questioning authority",
    },
    {
        "name": "The Lovers",
        "meaning": "choice, alignment, love, partnership, and values",
        "reversed_meaning": "disharmony, misalignment, indecision, or divided priorities",
    },
    {
        "name": "The Chariot",
        "meaning": "determination, momentum, discipline, and victory through control",
        "reversed_meaning": "lack of direction, aggression, delay, or losing control",
    },
    {
        "name": "Strength",
        "meaning": "courage, patience, compassion, resilience, and inner power",
        "reversed_meaning": "self-doubt, forcefulness, fear, or depleted confidence",
    },
    {
        "name": "The Hermit",
        "meaning": "solitude, reflection, wisdom, and searching within",
        "reversed_meaning": "isolation, withdrawal, loneliness, or avoiding guidance",
    },
    {
        "name": "Wheel of Fortune",
        "meaning": "cycles, luck, change, fate, and turning points",
        "reversed_meaning": "resistance to change, bad timing, setbacks, or repeated patterns",
    },
    {
        "name": "Justice",
        "meaning": "truth, fairness, accountability, and balanced decisions",
        "reversed_meaning": "dishonesty, bias, imbalance, or avoiding consequences",
    },
    {
        "name": "The Hanged Man",
        "meaning": "pause, surrender, new perspective, and willing sacrifice",
        "reversed_meaning": "stalling, resistance, needless delay, or refusing to let go",
    },
    {
        "name": "Death",
        "meaning": "endings, transformation, release, and renewal",
        "reversed_meaning": "fear of change, stagnation, clinging, or delayed transition",
    },
    {
        "name": "Temperance",
        "meaning": "balance, moderation, harmony, patience, and integration",
        "reversed_meaning": "excess, imbalance, impatience, or conflicting priorities",
    },
    {
        "name": "The Devil",
        "meaning": "attachment, temptation, material focus, and confronting limitations",
        "reversed_meaning": "release, awareness, reclaiming power, or breaking unhealthy patterns",
    },
    {
        "name": "The Tower",
        "meaning": "sudden change, disruption, revelation, and necessary collapse",
        "reversed_meaning": "avoided disaster, fear of upheaval, delayed change, or quiet rebuilding",
    },
    {
        "name": "The Star",
        "meaning": "hope, healing, inspiration, renewal, and spiritual clarity",
        "reversed_meaning": "discouragement, loss of faith, fatigue, or dimmed optimism",
    },
    {
        "name": "The Moon",
        "meaning": "dreams, uncertainty, illusion, emotion, and the subconscious",
        "reversed_meaning": "clarity emerging, confusion lifting, fear, or hidden truth surfacing",
    },
    {
        "name": "The Sun",
        "meaning": "joy, success, vitality, openness, and confidence",
        "reversed_meaning": "temporary doubt, muted joy, unrealistic optimism, or delayed success",
    },
    {
        "name": "Judgement",
        "meaning": "awakening, reflection, reckoning, and answering a call",
        "reversed_meaning": "self-doubt, avoidance, harsh judgment, or ignoring a call",
    },
    {
        "name": "The World",
        "meaning": "completion, wholeness, achievement, and integration",
        "reversed_meaning": "unfinished business, delay, lack of closure, or a final step still needed",
    },
]


class TarotCardTool(Tool):
    def __init__(self):
        super().__init__(
            name="draw_tarot_card",
            description="randomly draw one card from the Tarot Major Arcana",
        )

    def run(self, parameters: dict[str, Any]) -> str:
        allow_reversed = parameters.get("allow_reversed", True)
        include_meaning = parameters.get("include_meaning", True)

        card = random.choice(MAJOR_ARCANA)
        is_reversed = bool(allow_reversed) and random.choice([False, True])
        orientation = "reversed" if is_reversed else "upright"

        if not include_meaning:
            return f"{card['name']} ({orientation})"

        meaning_key = "reversed_meaning" if is_reversed else "meaning"
        return f"{card['name']} ({orientation}): {card[meaning_key]}"

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="allow_reversed",
                type="boolean",
                description="whether the drawn card can appear reversed",
                required=False,
                default=True,
            ),
            ToolParameter(
                name="include_meaning",
                type="boolean",
                description="whether to include a brief interpretation for the drawn card",
                required=False,
                default=True,
            ),
        ]

def get_registry() -> ToolRegistry:
    """Get a tool registry containing the tool."""
    registry = ToolRegistry()
    registry.register_tool(TarotCardTool())

    return registry