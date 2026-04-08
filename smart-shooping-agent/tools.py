"""
工具模块：定义 Agent 可调用的所有工具

工具设计原则（Day 4 核心学习点）：
1. 每个工具做一件事，让 LLM 来组合
2. 工具之间有调用顺序：搜索 → 详情 → 对比 → 计算
3. docstring 写给 LLM 看，要明确"什么时候该用这个工具"
4. 返回值是格式化字符串，LLM 容易理解
5. 工具内部要有错误处理，返回有用的错误提示而非崩溃
"""

from langchain_core.tools import tool

# ============================================================
# 模拟商品数据库
# ============================================================
PRODUCTS = {
    "SKU001": {
        "id": "SKU001", "name": "Nike Air Zoom Pegasus 41",
        "category": "运动鞋", "price": 899, "original_price": 1099,
        "rating": 4.7, "stock": 156, "brand": "Nike",
        "features": ["透气网面", "Zoom Air 气垫", "适合日常跑步"],
        "description": "耐克经典跑鞋，React泡棉中底，鞋重280g，适合配速5'30\"-7'/km慢跑",
    },
    "SKU002": {
        "id": "SKU002", "name": "Adidas Ultraboost Light",
        "category": "运动鞋", "price": 1199, "original_price": 1499,
        "rating": 4.5, "stock": 78, "brand": "Adidas",
        "features": ["Boost 中底", "Primeknit 鞋面", "适合长距离跑步"],
        "description": "阿迪轻量跑鞋仅275g，Light Boost泡棉回弹出色，Continental大底抓地力强",
    },
    "SKU003": {
        "id": "SKU003", "name": "Sony WF-1000XM5",
        "category": "耳机", "price": 1699, "original_price": 1999,
        "rating": 4.6, "stock": 89, "brand": "Sony",
        "features": ["主动降噪35dB", "LDAC高清编码", "8小时续航"],
        "description": "索尼旗舰降噪耳机，V2处理器，地铁降噪约35dB，快充5分钟用60分钟",
    },
    "SKU004": {
        "id": "SKU004", "name": "HUAWEI FreeBuds Pro 3",
        "category": "耳机", "price": 1099, "original_price": 1299,
        "rating": 4.5, "stock": 234, "brand": "HUAWEI",
        "features": ["智能降噪47dB", "L2HC高清编码", "6.5小时续航"],
        "description": "华为旗舰降噪耳机，麒麟A2芯片，降噪深度47dB，IP54防水",
    },
    "SKU005": {
        "id": "SKU005", "name": "Samsonite 新秀丽商务双肩包",
        "category": "背包", "price": 459, "original_price": 699,
        "rating": 4.4, "stock": 312, "brand": "Samsonite",
        "features": ["15.6寸电脑仓", "防泼水面料", "USB充电口"],
        "description": "CORDURA面料商务背包25L，减震电脑仓，S型透气背板，净重0.98kg",
    },
    "SKU006": {
        "id": "SKU006", "name": "HUAWEI WATCH GT 5 Pro",
        "category": "手表", "price": 2488, "original_price": 2988,
        "rating": 4.7, "stock": 67, "brand": "HUAWEI",
        "features": ["ECG心电分析", "100+运动模式", "14天续航"],
        "description": "华为高端运动手表，1.43寸AMOLED，支持高尔夫球场地图和自由潜水模式",
    },
    "SKU007": {
        "id": "SKU007", "name": "IQUNIX OG80 虫洞 机械键盘",
        "category": "键盘", "price": 799, "original_price": 999,
        "rating": 4.8, "stock": 42, "brand": "IQUNIX",
        "features": ["Gasket结构", "三模热插拔", "PBT键帽"],
        "description": "Gasket结构机械键盘，蓝牙/2.4G/有线三模连接，热插拔轴座，RGB背光",
    },
}

# ============================================================
# 工具 1: 商品搜索（入口工具，通常最先被调用）
# ============================================================
@tool
def search_products(
        category: str = "",
        max_price: float = 0,
        min_rating: float = 0,
        keyword: str = "",
) -> str:
    """搜索商品列表。根据分类、价格、评分、关键词筛选商品。
    这是查找商品的第一步。返回的结果包含商品ID，可以用 get_product_detail 获取详情。
    Args:
            category: 商品分类。可选值：运动鞋、耳机、背包、手表、键盘。留空搜索全部
            max_price: 价格上限（元）。0 表示不限
            min_rating: 最低评分（1-5）。0 表示不限
            keyword: 搜索关键词，匹配商品名称和描述
    """
    results = []
    for product in PRODUCTS.values():
        if category and product["category"] != category:
            continue
        if max_price and product["max_price"] != max_price:
            continue
        if min_rating > 0 and product["rating"] < min_rating:
            continue
        if keyword and keyword.lower() not in product["name"].lower() and keyword not in product["description"]:
            continue
        results.append(product)

    if not results:
        return f"未找到符合条件的商品（分类={category}, 最高价={max_price}, 最低评分={min_rating}, 关键词={keyword}）。建议放宽筛选条件重试。"

    output = f"找到 {len(results)} 个商品：\n"
    for p in results:
        discount = round((1 - p["price"] / p["original_price"]) * 100)
        output += (
            f"\n• {p['name']} [ID: {p['id']}]\n"
            f"  ¥{p['price']}（省{discount}%） | {p['rating']}分 | 库存{p['stock']}件\n"
        )
    output += "\n💡 使用 get_product_detail(product_id) 可查看某个商品的完整详情"
    return output















