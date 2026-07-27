[简体中文](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md) | [English](README.en.md)

# HanziStyleForge Fusion

Windows용 실험적 한자 글꼴 재구성 도구입니다. `target.ttf`에서 글꼴 스타일을 학습하고 `ref.otf`에서 한자 구조를 가져와 설치 가능한 TTF 글꼴을 생성합니다.

> 장시간 무인 실행을 위해 체크포인트 재개, 안전 중지, 자동 재시도를 지원합니다.

## 주요 기능

- `fonts/target.ttf`에서 전체 및 지역 글꼴 스타일을 학습합니다.
- `refs/ref.otf`의 기본 글리프가 포함하는 모든 한자를 재구성합니다.
- 중국 본토, 대만, 홍콩, 일본, 한국등 다양한 참조 글꼴을 사용할 수 있습니다.
- 대상 글꼴의 라틴 문자, 숫자, 기호, 가나, 한글 및 주요 OpenType 데이터를 가능한 한 유지합니다.
- 학습, 생성, 후보 선택, QA, 벡터화, 글꼴 빌드를 자동화합니다.

## 작동 방식

```text
target.ttf: 스타일
        +
ref.otf: 한자 구조와 범위
        ↓
Style Encoder → VQ → Diffusion → Refiner / Retrieval / IDS
        ↓
후보 선택 → QA → 윤곽선 변환 → TTF
```

프로그램은 어느 지역 자형이 더 올바른지 판단하지 않습니다. 최종 한자 구조는 `ref.otf`의 기본 Unicode `cmap` 글리프를 따릅니다.

## 요구 사항

- Windows 11 64-bit
- CUDA를 지원하는 NVIDIA GPU
- Python 3.10 이상
- 최소 150 GB의 여유 디스크 공간 권장

입력 글꼴:

```text
fonts\target.ttf
refs\ref.otf
```

정적 글꼴 사용을 권장합니다. `target.ttf`에는 TrueType `glyf` 테이블이 있어야 합니다. `ref.otf`는 정적 TrueType 또는 정적 CFF OTF를 사용할 수 있습니다. 가변 글꼴, TTC, OTC는 지원하지 않습니다.

## 빠른 시작

1. 저장소를 다운로드하거나 복제합니다.
2. 스타일 원본 글꼴을 `fonts\target.ttf`에 넣습니다.
3. 구조 참조 글꼴을 `refs\ref.otf`에 넣습니다.
4. 환경을 설치합니다.

   ```text
   install_cuda130.bat
   ```

5. 프로젝트를 확인합니다.

   ```text
   verify_project.bat
   ```

6. 전체 파이프라인을 시작하거나 재개합니다.

   ```text
   run_months_resilient.bat
   ```


7. 안전 중지 요청:

   ```text
   request_safe_stop.bat
   ```

8. 재개하기 전에 중지 표시를 지웁니다.

   ```text
   clear_safe_stop.bat
   ```

## 출력

주요 출력:

```text
build\target-HanziStyleForge-Fusion.ttf
build\target-HanziStyleForge-Fusion.ttf.report.json
work_hanzistyleforge_fusion_months\qa\index.html
```

학습 데이터, 체크포인트, 생성 진행 상태는 다음 폴더에 저장됩니다.

```text
work_hanzistyleforge_fusion_months\
```

학습 중에는 이 폴더를 삭제하지 마십시오.

## 사용 전 확인 사항

- 전체 실행에는 며칠, 몇 주 또는 그 이상이 걸릴 수 있습니다.
- 저장소에는 글꼴 파일, 사전 학습 가중치 또는 타사 글꼴 데이터셋이 포함되지 않습니다.
- 생성 글꼴에는 `target.ttf`와 `ref.otf`의 라이선스가 모두 적용될 수 있습니다.
- 학습, 수정, 재배포 권한이 있는 글꼴만 사용하십시오.
- 이 프로젝트는 실험적입니다. 배포 전에 QA 페이지와 최종 글꼴을 직접 확인하십시오.

## 연구 및 참고 자료

HanziStyleForge Fusion은 독립 구현입니다. 다음 프로젝트와 논문은 아키텍처 설계에 참고되었습니다. 해당 프로젝트의 소스 코드, 사전 학습 가중치, 글꼴 데이터셋은 이 저장소에 포함되지 않습니다.

| 출처 | 참고한 방향 |
|---|---|
| [zi2zi](https://github.com/kaonashi-tyc/zi2zi) | 한자 스타일 변환, 내용과 스타일 분리 |
| [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) | 확산 생성, 다중 스케일 내용 집계, 명시적 스타일 제약 |
| [HanziGen](https://github.com/wangwenho/HanziGen) | VQ 표현과 조건부 잠재 확산 |
| [VQ-Font](https://github.com/Yaomingshuai/VQ-Font) | 이산 글꼴 token과 구조 인식 강화 |
| [LF-Font / MX-Font](https://github.com/clovaai/fewshot-font-generation) | 지역 부품 스타일, 인자 분해, 다중 전문가 |
| [DeepVecFont-v2](https://github.com/yizhiwang96/deepvecfont-v2) | Transformer 벡터 시퀀스와 윤곽선 보정 |
| [Efficient and Scalable Chinese Vector Font Generation via Component Composition](https://arxiv.org/abs/2404.06779) | 부품 영역 변환과 대규모 조합 |
| [cjkvi/cjkvi-ids](https://github.com/cjkvi/cjkvi-ids) | Unicode IDS 부품 구조와 지역 힌트 |

인용은 방법상의 참고만 의미하며, 상위 프로젝트의 코드, 가중치, 데이터 또는 글꼴을 복사할 권한을 부여하지 않습니다. 타사 자료를 사용하기 전에 현재 라이선스와 이용 약관을 확인하십시오.

[zi2zi-JiT](https://github.com/kaonashi-tyc/zi2zi-JiT)는 아래에 별도로 기재합니다. 아키텍처 참고를 넘어 선택적 생성 백엔드로 사용할 수 있기 때문입니다.

## 선택적 생성 백엔드: zi2zi-JiT

생성 단계는 교체 가능합니다. 기본 백엔드는 이 프로젝트가 자체 구현한 Style Encoder → VQ → Diffusion → Refiner입니다. 대안으로 생성을 [zi2zi-JiT](https://github.com/kaonashi-tyc/zi2zi-JiT)(사전 학습 가중치를 제공하는 픽셀 공간 확산 Transformer)에 위임할 수 있습니다. 이때도 후보 선별, IDS 부품 검증, QA, 정밀화, 윤곽선 변환, TTF 빌드 등 하위 공정은 모두 HanziStyleForge Fusion이 담당합니다.

zi2zi-JiT의 소스 코드와 가중치는 이 저장소에 포함되어 있지 않습니다. 상위 저장소 복제와 가중치 다운로드는 직접 하셔야 하며, 백엔드는 로컬 사본을 호출합니다.

### 사용법

백엔드는 `config.json`의 `backend` 블록에서 선택하며, `--backend`로 한 번만 덮어쓸 수 있습니다.

```text
hanzistyleforge.py --backend=zi2zi-jit fusion-generate
```

사용 가능한 값은 `native`(기본값, 이 프로젝트 자체 생성 스택), `zi2zi-jit`, 그리고 이미 생성된 이미지 디렉터리를 읽는 `dir`입니다. `dir`은 수동 생성 결과를 이어 붙이거나, 생성기에 의존하지 않고 후처리 공정만 검증할 때 유용합니다.

```json
"backend": {
  "name": "zi2zi-jit",
  "candidate_count": 3,
  "zi2zi_jit": {
    "repo_dir": "D:/zi2zi-JiT",
    "checkpoint": "D:/zi2zi-JiT/run/lora_target/checkpoint-last.pth",
    "font_label": 0
  }
}
```

`python_executable`을 비워 두면 HanziStyleForge를 실행 중인 인터프리터를 재사용합니다. zi2zi-JiT의 추론 경로에는 torch, numpy, opencv, einops만 필요하며 `environment.yaml`에 고정된 구성은 필요하지 않습니다.

### LoRA 미세 조정이 선행되어야 합니다

**공개된 JiT-B/16 가중치는 사전 학습 산출물이며 제로샷으로 사용할 수 없습니다.** 처음 보는 글꼴에 그대로 적용하면 획이 체계적으로 누락됩니다. zi2zi-JiT README의 생성 예시는 모두 미세 조정된 가중치를 사용합니다.

`scripts/generate_font_dataset.py`로 데이터셋을 만들되 소스 글꼴은 추론 시 사용할 `ref.otf`와 동일하게 지정하십시오. 사전 학습에 맞추는 것보다 추론 시 내용 분포에 맞추는 것이 더 중요합니다. 이어서 `lora_single_gpu_finetune_jit.py`를 실행하고, 결과 가중치를 `checkpoint`에 지정한 뒤 `font_label`을 `0`으로 설정합니다(단일 글꼴 데이터셋은 `001_<name>`으로 배치되기 때문입니다). `font_label`을 비워 두면 label-drop 토큰을 사용하는데, 이는 기본 가중치에서만 의미가 있습니다.

Windows에서는 추가로 `TORCHDYNAMO_DISABLE=1`(Triton의 Windows 빌드가 없음), 저장소 루트를 가리키는 `PYTHONPATH`(`scripts/` 아래 스크립트는 자기 디렉터리가 `sys.path[0]`이 됨), `--num_workers 0`(DataLoader 워커가 lambda를 포함한 dataset을 pickle해야 함), `--online_eval` 미사용(FID를 계산하는데 PyPI의 torch-fidelity가 상위 프로젝트가 쓰는 fork와 API가 다름)이 필요합니다.

### 백엔드에 별도의 토폴로지 기준을 두는 이유

전역 `topology` 임계값은 내장 생성기에 맞춰 보정되어 있습니다. 내장 생성기는 참조에 structure-lock되므로 그 골격을 매우 근접하게 따라갑니다. 실제 스타일 변환을 수행하는 백엔드는 설계상 그로부터 벗어나므로, 같은 임계값을 적용하면 모든 출력이 거부됩니다(실측 `topology_score` 중앙값 0.14, 상한 0.06).

그래서 `backend.topology`는 비 native 백엔드에 한해 골격 유사도 상한만 완화합니다. **완화하지 않는 것은 연결 요소, 구멍, 오일러 수의 차이**이며, 이 값들은 0으로 유지되어 생성된 글자가 같은 문자임을 보장합니다. 같은 실측에서 이들의 중앙값은 이미 0이었으므로 정상적인 스타일 변환은 통과하고, 획이 늘거나 줄어든 글자는 거부되어 참조로 대체됩니다.

> **저작자 표시 의무.** zi2zi-JiT의 코드는 MIT 라이선스이지만, "Font Artifact License Addendum"이 산출물에 추가 조건을 부과합니다. 그 출력으로 만든 문자가 **200자를 초과하는** 글꼴 제품을 배포하는 경우 출처를 표시해야 합니다. 이 도구의 일반적인 실행은 200자를 크게 넘으므로, 이 백엔드를 사용했다면 표시가 필요하다고 가정하십시오. "Created using zi2zi-JiT artifacts"를 명시하고 상위 저장소 링크를 첨부합니다. 기본 백엔드로 생성한 글꼴에는 적용되지 않습니다. 자세한 내용은 `THIRD_PARTY_NOTICES.md`를 참조하십시오.

## 기여

Issue와 Pull Request를 환영합니다. 타사 코드, 데이터 또는 모델을 추가할 때는 출처와 라이선스 정보를 함께 명시하십시오.
