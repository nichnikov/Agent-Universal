"""
State management for the Universal Autonomous Agent.
Defines the global state structure passed between graph nodes.
"""

from typing import Annotated, Sequence, TypedDict, Optional, Dict, List, Any
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """
    Global state for the agent graph.
    
    This state is passed between all nodes in the LangGraph workflow.
    """
    # История сообщений. operator.add обеспечивает append, а не перезапись
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Указатель на следующий шаг (имя узла или "FINISH")
    next: str

    # Результаты поиска для использования в промптах
    # {query: [{title, url, content}]}
    # Используем перезапись (не append), чтобы хранить только актуальные результаты последнего поиска
    search_results: Optional[Dict[str, List[Dict[str, Any]]]]
