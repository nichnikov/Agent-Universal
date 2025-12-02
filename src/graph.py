"""
Graph assembly - сборка LangGraph с узлами и ребрами.
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes.supervisor import supervisor_node
from .nodes.legal_expert import legal_expert_node
from .nodes.accounting_expert import accounting_expert_node


def create_agent_graph():
    """
    Создает и компилирует граф агентов.
    
    Returns:
        Compiled LangGraph application
    """
    # Создаем граф с типом состояния AgentState
    workflow = StateGraph(AgentState)
    
    # Добавляем узлы
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("legal_expert", legal_expert_node)
    workflow.add_node("accounting_expert", accounting_expert_node)
    
    # Определяем функцию для условного ребра от supervisor
    def route_supervisor(state: AgentState) -> Literal["legal_expert", "accounting_expert", "__end__"]:
        """
        Маршрутизация от supervisor на основе поля 'next' в состоянии.
        
        Args:
            state: Текущее состояние агента
            
        Returns:
            Имя следующего узла или END
        """
        next_step = state.get("next", "FINISH")
        
        if next_step == "LegalExpert":
            return "legal_expert"
        elif next_step == "AccountingExpert":
            return "accounting_expert"
        else:
            # Для всех остальных случаев (включая "FINISH") завершаем
            return "__end__"
    
    # Добавляем ребра
    # От экспертов всегда возвращаемся к supervisor для проверки завершения
    workflow.add_edge("legal_expert", "supervisor")
    workflow.add_edge("accounting_expert", "supervisor")
    
    # От supervisor используем условное ребро
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "legal_expert": "legal_expert",
            "accounting_expert": "accounting_expert",
            "__end__": END
        }
    )
    
    # Устанавливаем точку входа
    workflow.set_entry_point("supervisor")
    
    # Компилируем граф
    return workflow.compile()


# Создаем глобальный экземпляр приложения
app = create_agent_graph()
