from app.models.user import BeautyProfile
from app.services.embedding_service import EmbeddingService
import numpy as np


class UserTowerService:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
    
    def profile_to_text(self, profile: BeautyProfile) -> str:
        """뷰티 프로필을 자연어 텍스트로 변환"""
        text_parts = []
        text_parts.append(f"{profile.skin_type.value} 피부 타입입니다.")
        text_parts.append(f"{profile.skin_tone.value} 피부 톤입니다.")

        if profile.concerns:
            concern_text = ", ".join(profile.concerns)
            text_parts.append(f"저의 피부 고민은 {concern_text}입니다.")
        
        if profile.has_allergy and profile.allergies:
            allergy_text = ", ".join(profile.allergies)
            text_parts.append(f"알레르기가 있으며, {allergy_text}에 알레르기가 있어 피해야 합니다.")

        if profile.interest_categories:
            category_text = ", ".join(profile.interest_categories)
            text_parts.append(f"{category_text} 제품에 관심이 많습니다.")

        text_parts.append(f"가격대는 {profile.min_price}원에서 {profile.max_price}원 사이를 선호합니다.")

        if profile.additional_info:
            text_parts.append(profile.additional_info.strip())

        return " ".join(text_parts)

    def generate_user_embedding(self, profile: BeautyProfile) -> np.ndarray:
        """뷰티 프로필을 임베딩 벡터로 변환"""
        profile_text = self.profile_to_text(profile)
        embedding = self.embedding_service.encode_text(profile_text)
        return np.array(embedding)
        