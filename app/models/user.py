from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class SkinType(str, Enum):
    OILY = "지성"
    DRY = "건성"
    SENSITIVE = "민감성"
    COMBINATION = "복합성"
    VERY_DRY = "악건성"
    TROUBLE = "트러블성"
    NEUTRAL = "중성"


class SkinTone(str, Enum):
    WARM = "웜톤"
    COOL = "쿨톤"
    NEUTRAL = "뉴트럴톤"
    SPRING_WARM = "봄웜톤" 
    SUMMER_COOL = "여름쿨톤" 
    AUTUMN_WARM = "가을웜톤"
    WINTER_COOL = "겨울쿨톤"


class BeautyProfile(BaseModel):
    skin_type: SkinType
    skin_tone: SkinTone
    concerns: List[str]
    has_allergy: bool
    allergies: Optional[List[str]] = []
    interest_categories: List[str]
    min_price: int
    max_price: int
    additional_info: str