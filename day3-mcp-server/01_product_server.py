"""
Day 3 - 淘天商品导购 MCP Server
目标：实现 Tool / Resource / Prompt 三类 MCP 能力

这个 Server 模拟了一个"淘天商品目录"服务，暴露以下能力：
  Tools:     search_products, get_product_detail, compare_products
  Resources: product://catalog (商品列表), product://policies (退换货政策)
  Prompts:   shopping_guide (导购话术模板), product_comparison (商品对比模板)

运行方式：
  方式1（MCP Inspector 测试）: uv run mcp dev day3-mcp-server/01_product_server.py
  方式2（stdio 模式）:          uv run python day3-mcp-server/01_product_server.py

关键理解：
  - FastMCP 是官方 Python SDK 提供的高级 API，用装饰器定义能力
  - 和 Day 1 的 @tool 装饰器类似，但遵循 MCP 协议标准
  - 任何 MCP 兼容的 Client（Claude Desktop, Cursor 等）都能直接连你的 Server
"""

from mcp.server.fastmcp import FastMCP

# ============================================================
# 初始化 MCP Server
# ============================================================
# 参数 "product-catalog" 是这个 Server 的名称，Client 连接时会看到
mcp = FastMCP("product-catalog")

# ============================================================
# 模拟商品数据库（和 Day 1 类似，但这次通过 MCP 协议暴露）
# ============================================================

PRODUCTS = {
    "SKU001": {
        "id": "SKU001",
        "name": "Nike Air Zoom Pegasus 41",
        "category": "运动鞋",
        "price": 899,
        "original_price": 1099,
        "rating": 4.7,
        "stock": 156,
        "features": ["透气网面", "Zoom Air 气垫", "适合日常跑步训练"],
        "description": "耐克旗下最经典的日常训练跑鞋，前掌和后跟各配一块独立气垫单元，React泡棉中底提供轻量缓震。鞋重约280g，适合配速5'30\"-7'/公里的慢跑训练。",
    },
    "SKU002": {
        "id": "SKU002",
        "name": "Adidas Ultraboost Light",
        "category": "运动鞋",
        "price": 1199,
        "original_price": 1499,
        "rating": 4.5,
        "stock": 78,
        "features": ["Boost 中底", "Primeknit 鞋面", "适合长距离跑步"],
        "description": "阿迪达斯2024年轻量化跑鞋，整鞋仅275g。全掌Light Boost泡棉密度降低30%但回弹提升5%，Continental马牌橡胶大底湿地抓地力优异。",
    },
    "SKU003": {
        "id": "SKU003",
        "name": "Sony WF-1000XM5",
        "category": "耳机",
        "price": 1699,
        "original_price": 1999,
        "rating": 4.6,
        "stock": 89,
        "features": ["主动降噪", "LDAC 高清编码", "8小时续航"],
        "description": "索尼旗舰降噪耳机，V2处理器降噪提升20%，地铁环境降噪约35dB。支持LDAC 990kbps传输，单次续航8小时，快充5分钟用60分钟。",
    },
    "SKU004": {
        "id": "SKU004",
        "name": "HUAWEI FreeBuds Pro 3",
        "category": "耳机",
        "price": 1099,
        "original_price": 1299,
        "rating": 4.5,
        "stock": 234,
        "features": ["智能降噪", "L2HC 高清编码", "6.5小时续航"],
        "description": "华为旗舰降噪耳机，麒麟A2芯片，最大降噪深度47dB。支持L2HC 4.0编码1.5Mbps超高码率，IP54防尘防水。",
    },
    "SKU005": {
        "id": "SKU005",
        "name": "Samsonite 新秀丽商务双肩包",
        "category": "背包",
        "price": 459,
        "original_price": 699,
        "rating": 4.4,
        "stock": 312,
        "features": ["15.6寸电脑仓", "防泼水面料", "USB充电口", "减压背带"],
        "description": "CORDURA弹道尼龙面料商务背包，25L容量，净重0.98kg。独立电脑隔层减震设计，S型透气背板，隐藏式防盗口袋。",
    },
}


# ============================================================
# 能力一：Tools（LLM 调用的函数，类似 POST 接口）
# ============================================================
# Tools 是 LLM 能主动调用的功能。当用户问"有什么耳机推荐"时，
# LLM 会决定调用 search_products 工具来获取信息。

@mcp.tool()
def search_products(category: str = "", max_price: float = 0, keyword: str = "") -> str:
    """搜索商品。可按分类、最高价格、关键词筛选。
    Args:
            category: 商品分类，如"运动鞋"、"耳机"、"背包"。留空搜索全部
            max_price: 价格上限（元），0表示不限
            keyword: 搜索关键词，会匹配商品名称和描述
    """
    results = []
    for sku, product in PRODUCTS.items():
        if category and product["category"] != category:
            continue
        if max_price > 0 and product["price"] > max_price:
            continue
        if keyword and keyword not in product["name"] and keyword not in product["description"]:
            continue
        results.append(product)

    if not results:
        return f"未找到符合条件的商品（分类={category}, 最高价={max_price}, 关键词={keyword}）"

    output = f"找到 {len(results)} 个商品：\n"
    for p in results:
        discount = round((1 - p["price"] / p["original_price"]) * 100)
        output += f"\n - {p['name']} (ID: {p['id']})\n"
        output += f"  价格: ¥{p['price']} (原价¥{p['original_price']}, 省{discount}%)\n"
        output += f"  评分: {p['rating']}/5.0 | 库存: {p['stock']}件\n"

    return output

@mcp.tool()
def get_product_detail(product_id: str) -> str:
    """获取单个商品的详细信息。
        Args:
            product_id: 商品ID，如 SKU001、SKU003
    """
    product = PRODUCTS.get(product_id)
    if not product:
        available = ",".join(PRODUCTS.keys)
        return f"商品ID '{product_id}' 不存在。可用的ID: {available}"

    discount = round((1 - product["price"] / product["original_price"]) * 100)
    return (
        f"商品详情: {product['name']}\n"
        f"ID: {product['id']}\n"
        f"分类: {product['category']}\n"
        f"价格: ¥{product['price']} (原价¥{product['original_price']}, 优惠{discount}%)\n"
        f"评分: {product['rating']}/5.0\n"
        f"库存: {product['stock']}件\n"
        f"卖点: {', '.join(product['features'])}\n"
        f"详细描述: {product['description']}"
    )

@mcp.tool()
def compare_product(product_id_a: str, product_id_b: str) -> str:
    """对比两个商品的详细参数。
    Args:
            product_id_a: 第一个商品ID
            product_id_b: 第二个商品ID
    """
    a = PRODUCTS.get(product_id_a)
    b = PRODUCTS.get(product_id_b)

    if not a:
        return f"商品 {product_id_a} 不存在"
    if not b:
        return f"商品 {product_id_b} 不存在"

    price_diff = abs(a["price"] - b["price"])
    cheaper = a["name"] if a["price"] < b["price"] else b["name"]

    return (
        f"=== 商品对比 ===\n\n"
        f"【{a['name']}】\n"
        f"  价格: ¥{a['price']} | 评分: {a['rating']} | 库存: {a['stock']}件\n"
        f"  卖点: {', '.join(a['features'])}\n\n"
        f"【{b['name']}】\n"
        f"  价格: ¥{b['price']} | 评分: {b['rating']} | 库存: {b['stock']}件\n"
        f"  卖点: {', '.join(b['features'])}\n\n"
        f"差价: ¥{price_diff} ({cheaper}更便宜)"
    )

# ============================================================
# 能力二：Resources（应用程序读取的数据，类似 GET 接口）
# ============================================================
# Resources 是暴露给 Client 应用的只读数据。
# 和 Tools 的区别：Tools 由 LLM 决定何时调用，Resources 由应用程序主动读取。
# 比如：应用启动时读取商品目录作为上下文，这就是 Resource。

@mcp.resource("product://catalog")
def get_product_catalog() -> str:
    """获取完整的商品目录列表。
    这是一个只读资源，Client 应用可以在启动时读取，
    把商品概览加载到 LLM 的上下文中。
    """
    catalog = "=== 淘天自营商品目录 ===\n\n"
    for sku, product in PRODUCTS.items():
        catalog += f"[{sku}] {product['name']} - ¥{product['price']} ({product['category']})\n"
    catalog += f"\n共 {len(PRODUCTS)} 个商品"
    return catalog

@mcp.resource("product://policies")
def get_policies() ->str:
    """获取退换货和售后政策。
        包含七天无理由退换、质量问题退换、价格保护等政策说明。
    """
    return """=== 淘天自营售后政策 ===
    1. 七天无理由退换
       自签收之日起7天内，商品未使用且不影响二次销售，可无理由退换。退货运费平台承担。
    2. 质量问题退换
       签收15天内发现质量问题，可退换货，运费平台承担。超15天在保修期内可联系品牌售后。
    3. 价格保护
       购买后15天内降价可申请差价补偿。大促期间（双11/618）延长至30天。
    4. 正品保障
       所有自营商品均为品牌正品，支持官方验证，假一赔十。
    5. 物流时效
       一线城市24小时送达，二线48小时，三线及以下72小时。"""

# ============================================================
# 能力三：Prompts（用户可选的交互模板）
# ============================================================
# Prompts 是预定义的交互模板，用户可以在 Client 中选择使用。
# 比如：用户选择"导购模式"，Client 会用这个 Prompt 来引导 LLM 的行为。
@mcp.prompt()
def shopping_guide(user_need: str) -> str:
    """导购助手话术模板。帮助用户找到最合适的商品。
        Args:
            user_need: 用户的购物需求描述，如"想买跑步鞋，预算1000以内"
    """
    return f"""你是淘天自营平台的专业导购助手。用户的需求是："{user_need}"

    请按照以下流程帮助用户：
    1. 分析用户的核心需求（品类、预算、使用场景、偏好）
    2. 使用 search_products 工具搜索符合条件的商品
    3. 使用 get_product_detail 获取候选商品的详细信息
    4. 如果有多个候选，使用 compare_products 做对比
    5. 给出推荐理由，包含：为什么适合用户、价格优势、用户评价

    注意：
    - 只推荐库存充足的商品
    - 要提及当前的优惠信息
    - 如果用户需求模糊，先追问细化"""

@mcp.prompt()
def product_comparison(product_a: str, product_b: str) -> str:
    """商品对比分析模板。结构化地对比两个商品。
        Args:
            product_a: 第一个商品名称或ID
            product_b: 第二个商品名称或ID
    """
    return f"""请对 "{product_a}" 和 "{product_b}" 进行全面对比分析。

    对比维度：
    1. 价格与性价比
    2. 核心参数与性能
    3. 用户评价与口碑
    4. 适用场景
    5. 当前优惠力度

    请使用 compare_products 工具获取数据，然后基于数据给出客观分析和最终推荐。
    如果两个商品面向不同场景，请说明各自的最佳适用人群。"""

# ============================================================
# 启动 Server
# ============================================================
if __name__ == "__main__":
    # 默认使用 stdio transport（标准输入输出）
    # 适合本地测试和 Claude Desktop 集成
    mcp.run()













