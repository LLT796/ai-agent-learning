"""
Day 3 - MCP Client 测试脚本
目标：用代码连接 MCP Server，验证 Tool / Resource / Prompt 三类能力

这个脚本会：
1. 启动 MCP Server（子进程，stdio 模式）
2. 作为 Client 连接到 Server
3. 依次测试 Tools、Resources、Prompts
4. 打印完整的交互日志

运行方式：uv run python day3-mcp-server/02_test_client.py

关键理解：
  - MCP Client 通过 stdio（标准输入输出）与 Server 通信
  - Client 先发现 Server 有哪些能力（list_tools/list_resources/list_prompts）
  - 然后按需调用（call_tool/read_resource/get_prompt）
  - 这个过程和 HTTP 的 API 发现 → 调用 非常类似
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    print("=" * 60)
    print("🔌 Day 3 - MCP Client 测试")
    print("=" * 60)

    # Server 脚本的路径
    server_script = str(Path(__file__).parent / "01_product_server.py")

    # 配置 stdio transport 通过子进程启动 Server
    server_params = StdioServerParameters(
        command=sys.executable, # 用当前 Python 解释器
        args=[server_script],
    )

    print(f"\n📡 正在连接 MCP Server: {server_script}")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化连接
            await session.initialize()
            print("   ✅ 连接成功!\n")

            # =============================================
            # 测试 1: 发现并调用 Tools
            # =============================================
            print("=" * 50)
            print("📋 测试 Tools（LLM 调用的函数）")
            print("=" * 50)

            # 1a. 发现有哪些 Tools
            tools_result = await session.list_tools()
            print(f"\n🔧 Server 暴露了 {len(tools_result.tools)} 个 Tools:")
            for tool in tools_result.tools:
                print(f"   - {tool.name}: {tool.description[:50]}...")

            # 1b. 调用 search_products Tool
            print(f"\n--- 调用 search_products (category='耳机') ---")
            result = await session.call_tool(
                "search_products",
                arguments={"category": "耳机"},
            )
            for content in result.content:
                print(content.text)

            # 1c. 调用get_product_detail Tool
            print(f"\n--- 调用 get_product_detail (product_id='SKU003') ---")
            result = await session.call_tool(
                "get_product_detail",
                arguments={"product_id": "SKU003"},
            )
            for content in result.content:
                print(content.text)

            # 1d. 调用 compare_products Tool
            print(f"\n--- 调用 compare_products (SKU003 vs SKU004) ---")
            result = await session.call_tool(
                "compare_products",
                arguments={"product_id_a": "SKU003", "product_id_b": "SKU004"}
            )
            for content in result.content:
                print(content.text)

            # =============================================
            # 测试 2: 读取 Resources
            # =============================================
            print(f"\n{'=' * 50}")
            print("📚 测试 Resources（应用读取的数据）")
            print("=" * 50)

            # 2a. 发现有哪些Resources
            resources_result = await session.list_resources()
            print(f"\n📦 Server 暴露了 {len(resources_result.resources)} 个 Resources:")
            for resource in resources_result.resources:
                print(f"   - {resource.uri}: {resource.name}")

            # 2b. 读取商品目录
            print(f"\n--- 读取 product://catalog ---")
            result = await session.read_resource("product://catalog")
            for content in result.contents:
                print(content.text)

            # 2c. 读取售后政策
            print(f"\n--- 读取 product://policies ---")
            result = await session.read_resource("product://policies")
            for content in result.contents:
                # 只打印前 200 个字符预览
                preview = content.text[:200]
                print(f"{preview}...")

            # =============================================
            # 测试 3: 获取 Prompts
            # =============================================
            print(f"\n{'=' * 50}")
            print("💬 测试 Prompts（交互模板）")
            print("=" * 50)
            # 3a. 发现有哪些 Prompts
            prompts_result = await session.list_prompts()
            print(f"\n📝 Server 暴露了 {len(prompts_result.prompts)} 个 Prompts:")
            for prompt in prompts_result.prompts:
                print(f"   - {prompt.name}: {prompt.description[:50]}...")

            # 3b. 获取导购模板
            print(f"\n--- 获取 shopping_guide 模板 ---")
            result = await session.get_prompt(
                "shopping_guide",
                arguments={"user_need": "想买个降噪耳机，预算1500以内"}
            )
            for msg in result.messages:
                print(f"  [{msg.role}]: {msg.content.text[:150]}...")

            # 3c. 获取对比模板
            print(f"\n--- 获取 product_comparison 模板 ---")
            result = await session.get_prompt(
                "product_comparison",
                arguments={"product_a": "Sony XM5", "product_b": "华为 FreeBuds Pro 3"},
            )
            for msg in result.messages:
                print(f"  [{msg.role}]: {msg.content.text[:150]}...")

    print(f"\n{'=' * 60}")
    print("✅ 所有测试完成!")
    print("=" * 60)

    # 打印总结
    print("""
    📝 MCP 三大能力总结:

    ┌─────────┬──────────────────┬───────────────────┬──────────────────┐
    │         │ Tools            │ Resources         │ Prompts          │
    ├─────────┼──────────────────┼───────────────────┼──────────────────┤
    │ 类比    │ POST 接口        │ GET 接口          │ 模板文件         │
    │ 谁控制  │ LLM 决定调用     │ 应用程序主动读取  │ 用户选择         │
    │ 用途    │ 执行操作/查询    │ 加载上下文数据    │ 引导 LLM 行为    │
    │ 示例    │ 搜索商品、下单   │ 商品目录、政策    │ 导购话术、对比   │
    │ 发现    │ list_tools()     │ list_resources()  │ list_prompts()   │
    │ 使用    │ call_tool()      │ read_resource()   │ get_prompt()     │
    └─────────┴──────────────────┴───────────────────┴──────────────────┘
        """)

if __name__ == "__main__":
    asyncio.run(main())