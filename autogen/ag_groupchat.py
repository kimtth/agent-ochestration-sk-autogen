import os
import asyncio
from dataclasses import dataclass
from autogen_core import (
    DefaultTopicId,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    default_subscription,
    message_handler,
)
from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


@dataclass
class GroupMessage:
    text: str


@default_subscription
class Moderator(RoutedAgent):
    def __init__(self, client):
        super().__init__("Moderator")
        self._sys = SystemMessage(
            content="You are the committee moderator. Invite specialist analysis."
        )
        self._client = client

    @message_handler
    async def on_proposal(self, message: GroupMessage, ctx) -> None:
        invite = (
            f"Proposal: {message.text}\nFundamental and Risk analysts, please comment."
        )
        # broadcast to both specialists
        await self.publish_message(GroupMessage(invite), DefaultTopicId())


@default_subscription
class FundamentalAnalyst(RoutedAgent):
    def __init__(self, client):
        super().__init__("FundamentalAnalyst")
        self._sys = SystemMessage(content="Provide a brief fundamental analysis.")
        self._client = client

    @message_handler
    async def on_discussion(self, message: GroupMessage, ctx) -> None:
        resp = await self._client.create(
            [
                self._sys,
                UserMessage(
                    content=message.text, role="user", source="FundamentalAnalyst"
                ),
            ]
        )
        print("FundamentalAnalyst:", resp.content)


@default_subscription
class RiskManager(RoutedAgent):
    def __init__(self, client):
        super().__init__("RiskManager")
        self._sys = SystemMessage(content="Provide a brief risk assessment.")
        self._client = client

    @message_handler
    async def on_discussion(self, message: GroupMessage, ctx) -> None:
        resp = await self._client.create(
            [
                self._sys,
                UserMessage(content=message.text, role="user", source="RiskManager"),
            ]
        )
        print("RiskManager:", resp.content)


async def main():
    client = AzureOpenAIChatCompletionClient(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        model="gpt-4o-mini",  # a name of OpenAI model (not clear meaning of this value when using Azure OpenAI)
    )
    runtime = SingleThreadedAgentRuntime()
    # register three agents on the same topic
    await Moderator.register(runtime, "moderator", lambda: Moderator(client))
    await FundamentalAnalyst.register(
        runtime, "fundamental", lambda: FundamentalAnalyst(client)
    )
    await RiskManager.register(runtime, "risk", lambda: RiskManager(client))

    runtime.start()
    # kick off group chat
    await runtime.publish_message(
        GroupMessage("Should we invest in TechCorp Inc.?"), DefaultTopicId()
    )
    await runtime.stop_when_idle()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())

"""
FundamentalAnalyst: To determine whether to invest in TechCorp Inc., a fundamental analysis should be conducted considering several key factors:

1. **Financial Health**:
   - **Revenue Growth**: Analyze the company's revenue trends over the past few years. Consistent revenue growth can indicate strong market demand for its products or services.
   - **Profit Margins**: Review gross and net profit margins. Higher margins suggest efficient operations and the ability to manage costs.
   - **Balance Sheet**: Examine assets versus liabilities. A strong balance sheet with manageable debt levels indicates financial stability.

2. **Valuation Metrics**:
   - **Price-to-Earnings (P/E) Ratio**: Compare TechCorp's P/E ratio with industry peers and historical averages. A lower P/E may suggest undervaluation.
   - **Price-to-Sales (P/S) and Price-to-Book (P/B) Ratios**: Similar comparisons can provide insights into the company's overall valuation.

3. **Market Position**:
   - **Competitive Advantage**: Identify whether TechCorp has a unique selling proposition or competitive edge (e.g., patents, technology, brand loyalty).
   - **Market Share**: Assess TechCorp's position in the tech industry regarding market share and growth potential.

4. **Industry Trends**:
   - **Growth Prospects**: Consider the overall growth outlook for the tech sector and how TechCorp fits into these trends (e.g., AI, cloud computing, cybersecurity).
   - **Regulatory Environment**: Stay informed about any regulations that could impact TechCorp's operations or growth prospects.

5. **Management Team**:
   - **Leadership Experience**: Evaluate the management team's background and track record in the tech industry.
   - **Strategic Vision**: Understand the company's roadmap, including product developments or market expansions.

6. **Economic Factors**:
   - **Macroeconomic Conditions**: Consider how broader economic indicators (interest rates, inflation, consumer confidence) could impact TechCorp's performance.

If these indicators suggest strong financial performance, reasonable valuation, growth potential, and effective management, TechCorp Inc. could be a solid investment option. However, individual risk tolerance and investment goals should also be considered. Always consult with a financial advisor for personalized advice tailored to your specific situation.
RiskManager: ### Risk Assessment for Investment in TechCorp Inc.

#### Fundamental Analysis
1. **Financial Performance**:
   - **Revenue Growth**: Analyze the company's recent revenue growth trends and projections. Consistent growth might indicate a strong market position.
   - **Profitability Metrics**: Review gross margin, operating margin, and net profit margin. High and improving margins could suggest efficient operations.
   - **Balance Sheet Strength**: Assess debt levels compared to equity. A manageable debt load indicates financial stability, while excessive debt could pose risks in downturns.

2. **Market Position**:
   - **Competitive Landscape**: Identify TechCorp’s market share and competitors. A strong positioning is beneficial but comes with risks from aggressive competitors.
   - **Innovation and R&D**: Evaluate their commitment to innovation. A robust pipeline can signal growth potential but also necessitates continual investment.

3. **Sector Trends**:
   - **Technology Adoption**: Consider trends influencing the tech sector, such as AI, cloud computing, or cybersecurity. These can provide growth opportunities but may also lead to obsolescence risks if TechCorp fails to adapt.

#### Risk Analysis
1. **Market Risk**:
   - **Volatility**: Tech stocks can be particularly volatile, influenced by broader market conditions and investor sentiment. Monitor economic indicators and market cycles.

2. **Regulatory Risk**:
   - **Compliance**: As a tech company, TechCorp may be subject to stringent regulations, particularly regarding data privacy and antitrust issues, which could impact operations and profitability.

3. **Operational Risk**:
   - **Supply Chain Dependencies**: If TechCorp relies on a global supply chain, disruptions (e.g., geopolitical tensions, pandemics) could impact production and delivery.
   - **Cybersecurity Threats**: As a tech entity, they face ongoing risks from data breaches or cyber-attacks which could damage brand reputation and financials.

4. **Technological Obsolescence**:
   - **Rapid Technology Changes**: The constant evolution in technology can lead to product obsolescence. TechCorp has to stay ahead or risk losing market relevance.

5. **Management and Governance Risk**:
   - **Leadership Stability**: Evaluate the management team’s experience and track record. Frequent changes in leadership can foreshadow instability.

#### Conclusion
Investing in TechCorp Inc. presents both opportunities and risks. Due diligence should focus on financial health, market dynamics, and operational vulnerabilities. A balanced approach that weighs potential returns against identified risks will be critical in making an informed investment decision. Further analysis and ongoing monitoring will be imperative post-investment.
FundamentalAnalyst: To undertake a fundamental analysis of TechCorp Inc., we need to assess several key aspects of the company's financial health, business model, industry position, and macroeconomic factors. Here's a brief overview based on common fundamental analysis principles:

1. **Financial Performance**:
   - **Revenue Growth**: Evaluate TechCorp's revenue growth over the last few years. A consistent upward trend in revenue is a positive indicator of market demand and company performance.
   - **Profit Margins**: Look at gross, operating, and net profit margins. High or improving margins may indicate efficient management and a strong competitive position.
   - **Earnings Per Share (EPS)**: Track EPS growth. Increasing EPS can signal strong profitability and can influence stock price positively.

2. **Balance Sheet Strength**:
   - **Debt Levels**: Assess the company's debt-to-equity ratio. A lower ratio is generally preferable as it indicates less financial risk. Be cautious if the company has high leverage, especially in a rising interest rate environment.
   - **Liquidity**: The current ratio and quick ratio will provide insight into TechCorp’s ability to meet short-term obligations. Ideally, both ratios should be above 1.

3. **Valuation Metrics**:
   - **Price-to-Earnings (P/E) Ratio**: Compare TechCorp’s P/E ratio to industry averages and historical levels. A high P/E may suggest overvaluation, while a low P/E might indicate undervaluation or declining business prospects.
   - **Price-to-Sales (P/S) Ratio**: This can be important for tech firms with fluctuating profits. Comparisons with peers can provide context for valuation.

4. **Market Position and Competitive Advantage**:
   - Analyze TechCorp’s position within the tech industry. Does it have a strong brand, unique technology, or intellectual property? A sustainable competitive advantage is crucial for long-term success.

5. **Industry Trends and Economic Conditions**:
   - Assess broader industry trends, such as technological advancements and regulatory changes affecting the tech sector. How well is TechCorp positioned to adapt to these changes?
   - Consider macroeconomic factors like interest rates, inflation, and consumer spending that could impact TechCorp’s operations.

6. **Management and Governance**:
   - Evaluate the quality and experience of TechCorp’s management team. Strong leadership is vital for navigating challenges and capitalizing on opportunities.

7. **Risks**:
   - Identify potential risks such as market competition, regulatory changes, cybersecurity threats, and economic downturns. Understanding the risk profile will help in making an informed investment decision.        

**Conclusion**: Based on the analysis, you will need to weigh the potential for high growth against the risks involved. Investing in TechCorp could be attractive if the company demonstrates strong fundamentals and growth potential, but be mindful of the specific risks in the tech sector and the broader economic environment. A more thorough due diligence process would involve detailed examination of the latest financial statements, news reports, and industry analyses.
RiskManager: To conduct a risk assessment for investing in TechCorp Inc., several factors should be considered:

1. **Market Position**: Evaluate TechCorp's position within the technology sector. Are they a market leader, and do they have a competitive edge over their rivals?

2. **Financial Health**: Analyze financial statements to assess profitability, revenue growth, debt levels, and cash flow. Look for trends in these areas over recent quarters or years.

3. **Product Innovation**: Consider the company's track record for innovation and the pipeline for new products. Are they keeping up with technological advancements and consumer demand?

4. **Regulatory Risks**: Identify any potential regulatory hurdles impacting the tech industry, such as data privacy concerns or antitrust regulations that could affect TechCorp.

5. **Economic Conditions**: Take into account the current economic landscape, including inflation rates, interest rates, and overall consumer spending trends, which could impact TechCorp's performance.

6. **Market Volatility**: The tech sector is often subject to rapid shifts. Assess the potential volatility and how external factors such as geopolitical tensions or global supply chain issues can impact the stock.  

7. **Management and Leadership**: Evaluate the experience and track record of the executive team. Strong leadership is vital for navigating challenges and capitalizing on opportunities.

8. **Analyst Ratings**: Review analyst reports and stock ratings. Consistent positive or negative outlooks from multiple analysts can provide insights into market sentiment.

9. **Investor Sentiment**: Gauge current investor sentiment towards TechCorp, as market perceptions can influence stock prices.

Based on this analysis, you can quantify the risk associated with investing in TechCorp Inc. and make a more informed decision. It is advised to weigh both potential returns and the identified risks before proceeding with the investment.
"""
