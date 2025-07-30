## AI Agent Orchestration

🤖🤝🤖 AI agent orchestration patterns using `Semantic Kernel` and `AutoGen`. 

The user scenario supports investment decisions by agents to analyze data using each orchestration pattern.

## Usage

```python
poetry install
```

- Semantic Kernel examples in `semantic_kernel/`:
  - `sk_concurrent.py`, `sk_sequential.py`, `sk_groupchat.py`, `sk_handoff.py`, `sk_magnetic.py`
- AutoGen examples in `autogen/`:
  - `ag_concurrent.py`, `ag_sequential.py`, `ag_groupchat.py`, `ag_handoff.py`

### 🤝 Orchestration patterns

| Orchestrations | Description                                                                                                                                                                                     |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ⚡ Concurrent  | Useful for tasks that will benefit from independent analysis from multiple agents.                                                                                                              |
| 1️⃣ Sequential | Useful for tasks that require a well-defined step-by-step approach.                                                                                                                              |
| 🤝 Handoff     | Useful for tasks that are dynamic in nature and don't have a well-defined step-by-step approach.                                                                                                  |
| 💬 GroupChat   | Useful for tasks that will benefit from inputs from multiple agents and a highly configurable conversation flow.                                                                                |
| 🔮 Magentic    | GroupChat-like with a planner-based manager. Inspired by [Magentic One](https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/). |

### 🛠️ Implementation Details

**Scenario Mapping**:
- **Concurrent**: Parallel analysis by multiple independent specialists on a single investment request.  
- **Sequential**: Deterministic investment pipeline (data collection → fundamental analysis → report generation).  
- **GroupChat**: Investment committee simulation where a moderator and specialist members collaborate in a structured conversation.  
- **Handoff**: Investment query triage, routing client requests to an equity specialist or to a human advisor based on content.  
- **Magentic**: Planner generates a structured task plan, orchestrator dispatches tasks to specialists, then synthesizes the final recommendation.

### 👥 Agent List by Pattern
- **Concurrent**: FundamentalAnalyst, TechnicalAnalyst, SentimentAnalyst
- **Sequential**: DataCollector, FundamentalAnalyst, ReportGenerator
- **GroupChat**: Moderator, FundamentalAnalyst, RiskManager
- **Handoff** : TriageAdvisor, EquitySpecialist
- **Magentic**: Planner, Orchestrator

### 📚 Learn More

1. [AI agent orchestration patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)🏆
1. [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
1. [Autogen Documentation](https://microsoft.github.io/autogen)
1. [LlamaIndex Agentic Workflows](https://www.youtube.com/watch?v=72XxWkd8Jrk)
1. Official examples
    - [Autogen](https://github.com/microsoft/autogen): `python/docs/src/user-guide/core-user-guide/design-patterns`
    - [Semantic Kernel](https://github.com/microsoft/semantic-kernel): `python/samples/getting_started_with_agents/multi_agent_orchestration`