# 전처리 (Preprocessing)

## 스크립트
`processed/make_cwt_gadf.py`

## 파이프라인
```
원시 진동 신호 (.mat)
  → 슬라이딩 윈도우 (window_size=4096 samples, non-overlapping)
  → CWT (Morlet wavelet, 128 scales, 1~2000 Hz)
      - 각 도메인마다 fs 다름 → scale을 fs 기준으로 계산
      - 출력: [128 scales × 4096 time steps]
  → 시간축 평균 → CWT power spectrum [128,]
  → [-1, 1] 정규화
  → GADF 변환 → [128×128] PNG 이미지
```

## 출력 디렉토리
```
processed_gadf_fine_4096/
  {domain}/          # 400, 402, 404, 500, ..., 804
    normal/          # N{domain}_*.png
    anomaly/         # B{domain}_*.png, I{domain}_*.png, O{domain}_*.png, ...
```

## 도메인 구성
| 파일 그룹 | 실제 의미 | fs |
|----------|----------|----|
| 400/500/.../800 | 배치 1 | 24.93 kHz |
| 402/502/.../802 | 배치 2 | 24.22 kHz |
| 404/504/.../804 | 배치 3 | ~23.0 kHz |

5개 RPM 조건 × 3 배치 = **15개 도메인**

## 클래스 레이블
| 파일 prefix | fault_type | label |
|------------|-----------|-------|
| N | 0 (normal) | 0 |
| I | 1 (inner race) | 1 |
| O | 2 (outer race) | 1 |
| B | 3 (ball) | 1 |
| IB/IO/OB | 4 (compound) | 1 |

## 주의사항
- B400, IB400 없음 (400 RPM ball/compound 미수집) → **fold_400 평가 불가**
- fs가 도메인마다 달라 CWT scale을 각각 계산해야 같은 Hz 범위 커버
- 이미지는 학습 시 224×224로 resize (transforms.Resize)

## 실행 명령
```bash
python processed/make_cwt_gadf.py \
  --data_dir "HUST bearing dataset" \
  --save_dir processed_gadf_fine_4096 \
  --domain_mode fine \
  --window_size 4096 \
  --n_scales 128 \
  --freq_min 1.0 \
  --freq_max 2000.0 \
  --num_workers 8
```
