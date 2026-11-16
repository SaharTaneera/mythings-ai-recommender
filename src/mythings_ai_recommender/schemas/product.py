

from pydantic import BaseModel, model_validator


class Product(BaseModel):
    id: str
    merchant_id: str

    name_en: str | None = None
    name_ar: str | None = None

    category: str | None = None
    subcategory: str | None = None

    description_en: str | None = None
    description_ar: str | None = None

    @model_validator(mode="after")
    def validate_product_name(self):
        if not any(
            value and value.strip()
            for value in (self.name_en, self.name_ar)
        ):
            raise ValueError(
                "At least one product name is required"
            )

        return self

