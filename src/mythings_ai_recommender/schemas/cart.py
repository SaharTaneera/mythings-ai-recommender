
from pydantic import BaseModel, Field


class CartItem(BaseModel):
    product_id: str
    quantity: int = Field(default=1, ge=1)


class Cart(BaseModel):
    merchant_id: str
    items: list[CartItem] = Field(default_factory=list)
