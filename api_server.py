"""
COSMOS EMOTION - REST API 서버
================================

FastAPI 기반 REST API 서버

실행:
    uvicorn api_server:app --reload --port 8000

접속:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import time
import numpy as np


# ============================================================================
# API 모델
# ============================================================================

class AnalysisRequest(BaseModel):
    """분석 요청"""
    text: str = Field(..., description="분석할 텍스트", min_length=1)
    fps: int = Field(25, description="Timeline FPS", ge=1, le=60)
    propagation_iterations: int = Field(2, description="전파 반복 횟수", ge=0, le=5)
    enable_visualization: bool = Field(False, description="시각화 생성 여부")


class EmotionScore(BaseModel):
    """감정 점수"""
    name: str
    value: float


class LayerEmotion(BaseModel):
    """계층별 감정"""
    layer: str
    emotions: List[EmotionScore]
    intensity: float


class ResonanceInfo(BaseModel):
    """공명 정보"""
    channel: str
    pattern_count: int
    avg_strength: float
    avg_amplification: float


class AnalysisResponse(BaseModel):
    """분석 응답"""
    success: bool
    text: str
    processing_time_ms: float
    
    # 계층별 감정
    layer_emotions: List[LayerEmotion]
    
    # 공명 정보
    resonance: List[ResonanceInfo]
    total_amplification: float
    
    # 메타데이터
    metadata: Dict
    
    # 시각화 URL (옵션)
    visualization_urls: Optional[List[str]] = None


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    version: str
    uptime_seconds: float


class BatchRequest(BaseModel):
    """배치 분석 요청"""
    texts: List[str] = Field(..., min_items=1, max_items=100)
    fps: int = 25
    propagation_iterations: int = 2


class BatchResponse(BaseModel):
    """배치 분석 응답"""
    success: bool
    total_texts: int
    results: List[AnalysisResponse]
    total_processing_time_ms: float


# ============================================================================
# FastAPI 앱 초기화
# ============================================================================

app = FastAPI(
    title="COSMOS EMOTION API",
    description="차세대 감정 분석 API - 음악 이론 + 양방향 전파 + 5채널 공명",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
engine = None
start_time = time.time()


# ============================================================================
# 시작/종료 이벤트
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 엔진 초기화"""
    global engine
    
    print("="*70)
    print("COSMOS EMOTION API 서버 시작")
    print("="*70)
    
    try:
        from integrated_cosmos_system import IntegratedCOSMOSEngine
        
        engine = IntegratedCOSMOSEngine(
            use_konlpy=False,
            fps=25,
            propagation_iterations=2
        )
        
        print("✓ 엔진 초기화 완료")
        print("✓ API 준비 완료")
        print("\n접속: http://localhost:8000/docs")
        print("="*70)
        
    except Exception as e:
        print(f"⚠ 엔진 초기화 실패: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료"""
    print("\n" + "="*70)
    print("COSMOS EMOTION API 서버 종료")
    print("="*70)


# ============================================================================
# API 엔드포인트
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """루트 엔드포인트"""
    return HealthResponse(
        status="running",
        version="2.0.0",
        uptime_seconds=time.time() - start_time
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스체크"""
    return HealthResponse(
        status="healthy" if engine is not None else "unhealthy",
        version="2.0.0",
        uptime_seconds=time.time() - start_time
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_emotion(request: AnalysisRequest):
    """
    감정 분석 API
    
    - **text**: 분석할 텍스트 (필수)
    - **fps**: Timeline FPS (기본: 25)
    - **propagation_iterations**: 전파 반복 횟수 (기본: 2)
    - **enable_visualization**: 시각화 생성 여부 (기본: False)
    """
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Engine not initialized"
        )
    
    start = time.time()
    
    try:
        # 분석 실행
        result = engine.analyze(request.text)
        
        # 응답 생성
        response = _create_response(
            text=request.text,
            result=result,
            processing_time=time.time() - start,
            enable_visualization=request.enable_visualization
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/analyze/batch", response_model=BatchResponse)
async def batch_analyze(request: BatchRequest):
    """
    배치 감정 분석
    
    최대 100개의 텍스트를 한 번에 분석
    """
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Engine not initialized"
        )
    
    if len(request.texts) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 texts allowed"
        )
    
    start = time.time()
    results = []
    
    try:
        for text in request.texts:
            text_start = time.time()
            result = engine.analyze(text)
            
            response = _create_response(
                text=text,
                result=result,
                processing_time=time.time() - text_start,
                enable_visualization=False
            )
            
            results.append(response)
        
        return BatchResponse(
            success=True,
            total_texts=len(request.texts),
            results=results,
            total_processing_time_ms=(time.time() - start) * 1000
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


@app.get("/emotions")
async def get_emotion_list():
    """
    지원하는 감정 목록 반환
    """
    emotions = [
        "joy", "sadness", "anger", "fear", "disgust", "surprise",
        "neutral", "han", "jeong", "nunchi", "hyeontta", "menboong",
        "excitement", "calmness", "empathic_pain", "amusement",
        "confusion", "disappointment", "guilt", "shame", "pride",
        "relief", "hope", "despair", "nostalgia", "contempt",
        "envy", "gratitude"
    ]
    
    return {
        "total": len(emotions),
        "emotions": emotions,
        "categories": {
            "basic": emotions[:7],
            "korean": emotions[7:12],
            "complex": emotions[12:]
        }
    }


@app.get("/stats")
async def get_statistics():
    """
    시스템 통계
    """
    return {
        "uptime_seconds": time.time() - start_time,
        "engine_status": "ready" if engine is not None else "not_ready",
        "supported_languages": ["ko", "en"],
        "max_batch_size": 100,
        "features": {
            "morpheme_analysis": True,
            "bidirectional_propagation": True,
            "resonance_detection": True,
            "timeline_generation": True,
            "visualization": True
        }
    }


# ============================================================================
# 헬퍼 함수
# ============================================================================

def _create_response(
    text: str,
    result,
    processing_time: float,
    enable_visualization: bool = False
) -> AnalysisResponse:
    """
    분석 결과 → API 응답 변환
    """
    from bidirectional_propagation import Layer
    
    # 계층별 감정
    layer_emotions = []
    
    for layer, emotion in result.layer_emotions.items():
        emotion_dict = emotion.to_dict()
        
        # 상위 5개 감정
        top_emotions = sorted(
            emotion_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        layer_emotions.append(LayerEmotion(
            layer=layer.name,
            emotions=[
                EmotionScore(name=name, value=float(value))
                for name, value in top_emotions
            ],
            intensity=float(emotion.intensity())
        ))
    
    # 공명 정보
    resonance_info = []
    
    for channel, patterns in result.resonance_patterns.items():
        if patterns:
            avg_strength = float(np.mean([
                p.resonance_strength for p in patterns
            ]))
            avg_amp = float(np.mean([
                p.amplification for p in patterns
            ]))
            
            resonance_info.append(ResonanceInfo(
                channel=channel,
                pattern_count=len(patterns),
                avg_strength=avg_strength,
                avg_amplification=avg_amp
            ))
    
    # 시각화 (옵션)
    viz_urls = None
    if enable_visualization:
        viz_urls = [
            "/visualizations/timeline.png",
            "/visualizations/layer_flow.png",
            "/visualizations/resonance.png"
        ]
    
    return AnalysisResponse(
        success=True,
        text=text,
        processing_time_ms=processing_time * 1000,
        layer_emotions=layer_emotions,
        resonance=resonance_info,
        total_amplification=float(
            result.amplification['total_amplification']
        ),
        metadata=result.metadata,
        visualization_urls=viz_urls
    )


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("COSMOS EMOTION API 서버 실행 중...")
    print("접속: http://localhost:8000/docs")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
