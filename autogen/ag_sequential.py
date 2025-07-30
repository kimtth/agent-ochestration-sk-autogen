import os
from dataclasses import dataclass
from typing import Dict, Any

from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    message_handler,
    type_subscription,
)
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.models import SystemMessage, UserMessage
from dotenv import load_dotenv

load_dotenv()


@dataclass
class InvestmentTask:
    company: str
    data: Dict[str, Any]


@type_subscription(topic_type="DataCollector")
class DataCollectorAgent(RoutedAgent):
    def __init__(self, client):
        super().__init__("Data Collector")
        self._system = SystemMessage(content="Gather and validate financial data.")
        self._client = client

    @message_handler
    async def on_collect(self, message: InvestmentTask, ctx: MessageContext) -> None:
        result = await self._client.create(
            messages=[
                self._system,
                UserMessage(content=str(message.data), source="DataCollector"),
            ],
            cancellation_token=ctx.cancellation_token,
        )
        message.data["collected"] = result.content
        await self.publish_message(
            message, topic_id=TopicId("FundamentalAnalyst", source=self.id.key)
        )


@type_subscription(topic_type="FundamentalAnalyst")
class FundamentalAnalystAgent(RoutedAgent):
    def __init__(self, client):
        super().__init__("Fundamental Analyst")
        self._system = SystemMessage(content="Perform fundamental analysis.")
        self._client = client

    @message_handler
    async def on_analyze(self, message: InvestmentTask, ctx: MessageContext) -> None:
        result = await self._client.create(
            messages=[
                self._system,
                UserMessage(
                    content=message.data["collected"], source="FundamentalAnalyst"
                ),
            ],
            cancellation_token=ctx.cancellation_token,
        )
        message.data["analysis"] = result.content
        await self.publish_message(
            message, topic_id=TopicId("ReportGenerator", source=self.id.key)
        )


@type_subscription(topic_type="ReportGenerator")
class ReportGeneratorAgent(RoutedAgent):
    def __init__(self, client):
        super().__init__("Report Generator")
        self._system = SystemMessage(content="Produce the final investment report.")
        self._client = client

    @message_handler
    async def on_report(self, message: InvestmentTask, ctx: MessageContext) -> None:
        result = await self._client.create(
            messages=[
                self._system,
                UserMessage(content=message.data["analysis"], source="ReportGenerator"),
            ],
            cancellation_token=ctx.cancellation_token,
        )
        print(f"\nFinal Report for {message.company}:\n{result.content}")


async def run_sequential_scenario():
    runtime = SingleThreadedAgentRuntime()
    client = AzureOpenAIChatCompletionClient(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        model="gpt-4o-mini",  # a name of OpenAI model (not clear meaning of this value when using Azure OpenAI)
    )
    await DataCollectorAgent.register(
        runtime, "DataCollector", lambda: DataCollectorAgent(client)
    )
    await FundamentalAnalystAgent.register(
        runtime, "FundamentalAnalyst", lambda: FundamentalAnalystAgent(client)
    )
    await ReportGeneratorAgent.register(
        runtime, "ReportGenerator", lambda: ReportGeneratorAgent(client)
    )

    runtime.start()
    task = InvestmentTask(
        company="GreenTech Solutions Inc.",
        data={"sector": "Renewable Energy", "market_cap": "$5B"},
    )
    await runtime.publish_message(
        task, topic_id=TopicId("DataCollector", source="default")
    )
    await runtime.stop_when_idle()
    await client.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_sequential_scenario())

"""

Final Report for GreenTech Solutions Inc.:
# Final Investment Report: Renewable Energy Company

### Company Overview
- **Market Capitalization**: $5 Billion
- **Industry**: Renewable Energy

### 1. Industry Analysis
- **Sector Growth**: The renewable energy sector is experiencing robust growth, driven by increasing global demand for sustainable energy sources, favorable government policies, and advancements in solar, wind, and battery technologies. The International Energy Agency (IEA) projects that renewables will account for the majority of global power generation by 2025.
  
- **Market Position**: The company is positioned as a mid-tier player in the marketplace, competing effectively against larger firms while also exploiting niche markets.

- **Competitors**: Key competitors include **Company A** (Market Cap: $10 billion, growth rate: 15%), **Company B** (Market Cap: $4 billion, growth rate: 12%), and **Company C** (Market Cap: $3 billion, growth rate: 10%). The competitive landscape is intensifying, particularly in solar technology and energy storage solutions.

### 2. Financial Statements
- **Income Statement**: 
  - Last reported revenue: $800 million, 20% year-over-year growth.
  - Profit margin: 12%.
  - Net Income: $96 million, demonstrating a steady profit increase.

- **Balance Sheet**:
  - Current Ratio: 2.5 (indicating strong liquidity).
  - Debt-to-Equity Ratio: 0.5 (showing sound financial leverage).

- **Cash Flow Statement**:
  - Positive cash flow from operating activities at $150 million.
  - Cash flow from investing shows sustainable assessments with a focus on expansion.

### 3. Valuation Metrics
- **P/E Ratio**: 25, compared to the industry average of 30, suggesting the company may be undervalued.
- **P/B Ratio**: 3.0, reflecting a reasonable valuation relative to its book value.
- **EV/EBITDA**: 10, on par with industry averages, indicating that the company is valued fairly in terms of operational performance.

### 4. Growth Potential
- **Market Opportunities**: The company is exploring the offshore wind segment and has recently secured contracts for solar energy projects in emerging markets, which could significantly boost revenue.

- **Product Innovations**: Investment in battery storage solutions positions the company to benefit from the growing demand for energy storage. They have filed patents for innovative solar panel technology with higher efficiency.

### 5. Risks
- **Regulatory Risks**: Potential changes to government incentives for renewables could impact profitability.

- **Market Competition**: Increased competition from both established and emerging companies poses a risk to market share.

- **Operational Risks**: Supply chain disruptions and project execution delays could affect project timelines and costs.

### 6. Dividend and Share Buyback Analysis
- **Dividends**: The company pays a modest dividend with a payout ratio of 25%, indicating viable returns to shareholders along with reinvestment in growth.

- **Share Buybacks**: Recently initiated a $150 million buyback program, reflecting management's confidence in the company's future performance.   

### 7. Management Effectiveness
- **Leadership Experience**: The CEO has over 20 years of experience in the energy sector with a strong track record of successful project management and innovation.

- **Corporate Governance**: Adopted strong governance practices with a diverse board emphasizing accountability and sustainability.

### 8. Current Events and Sentiment
- **Market Sentiment**: Analyst sentiment remains positive, driven by recent contract wins and strategic partnerships. Company has been highlighted in recent articles for its expansion strategy.

- **Earnings Calls/Webinars**: Recent earnings call emphasized a strategic focus on cost management and sustainability initiatives.

### Conclusion
The renewable energy company shows promising potential, marked by strong financial metrics, consistent growth, effective management, and positive industry trends. However, investors should remain aware of competitive risks and regulatory changes. Given the current valuation metrics and market opportunities, the company represents a potentially worthwhile investment for those looking to capitalize on the renewable energy sector's expansion.

**Recommendation**: Consider initiating a position in the company while monitoring external market conditions and company-specific developments closely. Further detailed analysis on quarterly performance and strategic developments is recommended for ongoing assessment.
"""
