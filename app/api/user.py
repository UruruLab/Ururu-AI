from fastapi import APIRouter, HTTPException, Depends
from app.core.dependencies import get_user_tower_service
from app.models.user import BeautyProfile
from app.services.user_tower_service import UserTowerService

router = APIRouter()


@router.post("/profile-embedding")
async def create_user_embedding(
    profile: BeautyProfile,
    service: UserTowerService = Depends(get_user_tower_service)
):
    """사용자 뷰티 프로필을 임베딩 벡터로 변환합니다."""
    try:
        natural_text = service.profile_to_text(profile)
        embedding = service.generate_user_embedding(profile)

        return {
            "status": "success",
            "profile": profile.dict(),
            "natural_text": natural_text,
            "embedding_shape": embedding.shape,
            "embedding_sample": embedding[:5].tolist()  
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임베딩 생성 중 오류 발생: {str(e)}") from e
