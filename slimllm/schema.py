"""电商售后抽取契约；仅依赖 Pydantic 2，兼容 Python 3.11+。"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


OrderId = Annotated[
    str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
]
ProductName = Annotated[str, Field(min_length=1, max_length=120, pattern=r"\S")]
Reason = Annotated[str, Field(min_length=1, max_length=300, pattern=r"\S")]


class AfterSalesTicket(BaseModel):
    """七个键全部必填；未知实体为 null，未知分类使用 unknown/other。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    intent: Literal["aftersales_request", "complaint", "inquiry", "other"] = Field(
        description="售后办理请求、单纯投诉、咨询或其他；有明确办理请求优先标为 aftersales_request。"
    )
    order_id: OrderId | None = Field(
        description="原文明确提供的订单号，保留前导零；未提供为 null，不推断。"
    )
    product_name: ProductName | None = Field(
        description="原文中的商品名称片段；未提供为 null，不补充品牌或型号。"
    )
    issue_type: Literal[
        "quality", "damaged", "missing", "wrong_item", "logistics",
        "not_as_described", "no_reason", "other", "unknown",
    ] = Field(
        description="功能质量、破损、缺件、错发、物流、描述不符、明确无理由、其他或未知。"
    )
    priority: Literal["low", "normal", "high", "urgent"] = Field(
        description="人身/财产安全风险 urgent；明确催急 high；明确不急 low；其余 normal。"
    )
    requested_action: Literal[
        "refund", "return_refund", "exchange", "repair", "reship",
        "track_delivery", "other", "unknown",
    ] = Field(description="仅退款、退货退款、换货、维修、补发、查催物流、其他或未明确。")
    reason: Reason | None = Field(
        description="支持问题分类的最短完整原文连续片段；无明确原因则为 null，不生成解释。"
    )
