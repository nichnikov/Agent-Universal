"""
Accounting Expert Node - специализированный агент для бухгалтерских консультаций.
"""

from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig

from ..state import AgentState
from ..tools.action_search_tool import create_search_tool
from .base_expert import execute_expert_node, ToolArgs


class ToolRequest(BaseModel):
    tool_name: Literal["internal_knowledge_search"]
    tool_args: ToolArgs

class AgentAction(BaseModel):
    action: Literal["call_tool", "final_answer"]
    tool: Optional[ToolRequest] = None
    content: Optional[str] = None
    references: Optional[list[str]] = None


async def accounting_expert_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Accounting Expert узел - обрабатывает бухгалтерские запросы.
    """
    # Инициализируем поиск с pubdivid=1 (Glavbukh)
    action_search = create_search_tool(default_pubdivid=1)
    
    tools_map = {
        "internal_knowledge_search": action_search
    }
    
    return await execute_expert_node(
        state=state,
        config=config,
        prompt_name="accounting-expert-prompt",
        tools_map=tools_map,
        response_model=AgentAction
    )
