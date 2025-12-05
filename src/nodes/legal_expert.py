"""
Legal Expert Node - специализированный агент для юридических консультаций.
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig

from ..state import AgentState
from ..tools.action_search_tool import create_search_tool
from .base_expert import execute_expert_node, ToolArgs


class ToolRequest(BaseModel):
    tool_name: Literal["search_legal_code", "internal_knowledge_search"]
    tool_args: ToolArgs

class AgentAction(BaseModel):
    action: Literal["call_tool", "final_answer"]
    tool: Optional[ToolRequest] = None
    content: Optional[str] = None
    references: Optional[list[str]] = None


async def legal_expert_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Legal Expert узел - обрабатывает юридические запросы с использованием structured output.
    """
    # Инициализируем инструменты
    # Для юридического эксперта используем USS (1jur.ru)
    action_search = create_search_tool(default_pubdivid=13, system_alias="uss")
    
    tools_map = {
        "search_legal_code": action_search, # Мапим старое название на новый инструмент
        "internal_knowledge_search": action_search
    }
    
    return await execute_expert_node(
        state=state,
        config=config,
        prompt_name="legal-expert-prompt",
        tools_map=tools_map,
        response_model=AgentAction
    )
