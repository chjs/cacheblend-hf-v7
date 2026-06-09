# CompBlend Goal-1 구현 & 검증 계획 + 비판적 분석 (2026-06-09)

**Goal 1**: KVzip(오프라인 압축) + CacheBlend(HKVD 선택재계산)를 **게이트 없이** 결합 = "only-HKVD compblend".
importance는 **오프라인 prune에만** 사용, blend는 **순수 HKVD(deviation)**.
**제약**: 검증된 `fuse_selective` 무수정 재사용(새 fuse 함수 금지), 압축-어댑터 레이어만 추가.

---

## Part A — 구현 계획

### A.1 아키텍처 (데이터 흐름)
```
오프라인:  doc text → KVzipBackend.score() → CompressedChunk(full pre-RoPE K/V + importance, rate=0)
           → token_prune(budget, reduce, protect_first) → pruned CompressedChunk → save(disk)
온라인:    load CompressedChunk(s) → to_blend_inputs → (list[Chunk], KVStore)
           + fresh query chunk(precompute) → fuse_selective(recompute_ratio, force_last_chunk,
             force_chunk_starts) → greedy decode → F1
```
핵심: pruned 압축청크 = 토큰 적은 평범청크(KVStore 레이아웃 동일) → 기존 fuse_selective가 그대로 HKVD blend.

### A.2 구현 완료 (DONE, compblend 브랜치)
- `compress/base.py`,`__init__.py` (@baaef43,@a9cabfc): `CompressedChunk`(save/load/to),`CompressionBudget`,
  `token_prune(reduce=mean|max|ranknorm_max, protect_first)`,`reduce_importance`,`to_blend_inputs`,
  `CompressionBackend` ABC(`score()`).
- `fuse_selective`: `force_chunk_starts=N` 추가(각 청크 첫 N토큰 force-recompute). blend 로직 0줄 변경(forced_mask만 확장).
- 테스트(CPU,4.51.3): `test_compress_blend.py` — token_prune 형태/예산/reduce, ranknorm_max sink 보존,
  protect_first, force_chunk_starts, e2e(pruned docs+fresh query→fuse_selective). 전체 6 스위트 PASS.

### A.3 남은 구현 (NEXT)
1. `compress/kvzip.py` — `KVzipBackend.score()`: lazy `ModelKVzip`, k_proj/v_proj 훅으로 pre-RoPE K/V 캡처,
   `kv.score`→importance. compblend7 `backends/kvzip.py` 포팅하되 variant C(`compress_full_context`) 제외.
   B안: score()는 full chunk(rate=0) 반환, prune은 token_prune이 별도.
2. 오프라인 precompute 스크립트 — doc 코퍼스 압축→CompressedChunk save(chunk_id 키).
3. 온라인 blend 벤치마크 — load→prune(ratio)→blend(HKVD)→F1. compblend7 blend_musique_generic_kvzip.py 참고.

---

## Part B — 검증 계획

### B.1 기계적 정확성 (CPU, 대부분 DONE)
token_prune·glue·force_chunk_starts·ranknorm_max 단위 테스트. KVzipBackend는 mock으로 ABC 계약 확인.

### B.2 실모델 GPU (vast.ai pod, transformers 4.51.3 + flash_attn + KVzip repo)
- **S1 sanity**: KVzip score()가 valid importance [L,H_kv,T] 생성; **sink(첫 토큰)이 실제로 높은 importance를
  받는지** 확인(사용자 가설 검증); ranknorm_max가 그 sink을 실제로 보존하는지.
- **S2 F1 그리드 (핵심)**: kvzip_ratio × recompute_ratio. arms:
  - `full_prefill_all` (절대 천장, 압축·reuse 없음)
  - **`full_prefill_survivors`** (★압축 후 천장 — prune된 생존자 집합을 함께 full-prefill. compblend의
    올바른 비교 기준. 이게 있어야 압축 gap과 blending gap을 분리)
  - `full_reuse_kvzip` (압축 생존자 reuse, recompute 없음 — blending gap 바닥)
  - `compblend` (압축 생존자 + HKVD 선택재계산 — 본 제안)
  - force_last_chunk=True(realistic serving) 공통.
  - gap 분해:  압축 gap = full_prefill_all − full_prefill_survivors (KVzip 책임, HKVD 무관);
              blending gap = full_prefill_survivors − full_reuse_kvzip (HKVD가 닫아야 할 것).
- **S3 sink 처리 ablation**: reduce∈{mean,max,ranknorm_max} × force_chunk_starts∈{0,1,4} × protect_first∈{0,1}.
- 모델 Mistral-7B-Instruct-v0.2, 데이터 MuSiQue, token-F1.

### B.3 판정 기준 (pre-register) — ★C.1 정정 반영
HKVD는 **삭제된 KV를 회복하지 못한다**(압축 gap은 KVzip 책임). HKVD가 하는 건 **생존자들 사이의
cross-attention 복구**(blending gap). 따라서 올바른 비교는 `full_prefill_survivors` 기준:
- **make-or-break**: `compblend`가 **blending gap을 닫아 `full_prefill_survivors`에 근접**하는가
  (compblend ≫ full_reuse_kvzip, 그리고 compblend → full_prefill_survivors). full_prefill_all이 아님.
- **2차(압축 gap)**: full_prefill_survivors가 full_prefill_all의 몇 %인가 = KVzip 압축의 순손실(HKVD 무관).
- **선택위험**: 압축 하 HKVD deviation이 [[hkvd-failure-mechanism]]처럼 부호 뒤집혀 **잘못된 토큰을 고르는지**
  진단(random 선택 대조군 포함) — only-HKVD라 게이트 교정이 없어 이 위험 노출.
- **sink**: sink 처리(ranknorm_max/protect/force_chunk_starts) on/off가 F1을 유의하게 올리나.

---

## Part C — 비판적 자기분석 (이 계획의 약점)

### C.1 🔴 두 gap을 구분하라 — HKVD는 삭제된 KV를 회복하지 못한다 (★정정)
처음 표현("HKVD가 압축 KV를 회복")은 **부정확**했다. 정확히는 손실이 둘:
- **압축 gap** (evict된 토큰 정보): HKVD로 **회복 불가**. KVzip 책임(생존자가 문맥 재구성한다는 전제).
  = full_prefill_all − full_prefill_survivors.
- **blending gap** (생존자 K/V의 cross-attention 결핍): **HKVD가 닫을 수 있는 것**. HKVD는 생존자 일부를
  재계산해 생존자들끼리·쿼리와의 cross-attention을 복구. = full_prefill_survivors − full_reuse_kvzip.

따라서 make-or-break = **compblend가 blending gap을 닫아 `full_prefill_survivors`에 근접**하는가 (full_prefill_all이
아님). 위험: (i) "double sparsity"로 생존자만으론 cross-attention 복구가 불충분, (ii) **압축 하에서 HKVD의
deviation 선택 자체가 망가짐**([[hkvd-failure-mechanism]]: corr −1→+0.88, 잘못된 토큰 고름) — only-HKVD라
게이트 교정이 없어 노출. 둘 중 하나라도면 compblend가 full_reuse_kvzip을 못 이길 수 있음.

### C.2 🔴 sink 처리 정당화가 미검증 가정 위에 서 있다
ranknorm_max "sink 보존" 테스트는 **합성 importance**(손으로 만든 픽스처)였다. **실제 KVzip importance가
sink에 "어느 head에서 top-rank"를 주는지는 미검증**. 이 가정이 틀리면 ranknorm_max/protect_first의 sink 논리
전체가 공허. → B.2 S1에서 **실 importance로 먼저 확인**해야 하고, 안 그러면 S3 ablation은 무의미.

### C.3 🟠 통계적 엄밀성 부재 (과거 "anecdotal" 반복 위험)
직전 GPU 검증은 N=15 "일화적, 유의성검정 X"였다. 이 계획도 N·CI·유의성을 명시 안 함. F1은 noisy하고
([[loong-f1-diagnosis]], "Answer within 5 words" 핵), MuSiQue F1은 metric 민감. → **paired bootstrap CI +
충분한 N(≥150) + 핵심 비교 pre-register(다중비교 통제)**를 박지 않으면 또 anecdotal로 끝남.

### C.4 🟠 실험 그리드 조합 폭발 → fishing 위험
reduce{3} × force_chunk_starts{3} × protect_first{2} × kvzip_ratio{n} × recompute_ratio{m} = 수십~수백 셀.
N이 작으면 underpowered, 다중비교로 우연한 "승리" 양산. → 그리드를 **단계적으로**(먼저 S2 핵심 4-arm, 그다음
sink ablation은 best regime 1곳에서만) 좁히고 통제.

### C.5 🟠 우리가 재현하려는 compblend7이 버그 있는 베이스라인
compblend7엔 우리가 찾은 버그들(budget 무시·chunk_id 오라벨·prune이 라이브러리 밖 등)이 있다. **그 "시나리오
(아키텍처)"를 재현하는 것과 그 "숫자"를 재현하는 것은 다르다.** 숫자가 버그를 반영할 수 있으니, 우리 깨끗한
구현이 다른 숫자를 내도 그게 "틀린" 게 아닐 수 있음 — 비교 기준을 숫자가 아니라 시나리오/상대경향으로 둬야.

### C.6 🟡 위치 재압축(연속)을 확정했으나 미검증
연속 재압축은 KVzip 원본(원위치)과 기하가 다름(앞서 논의). gapped 대안을 **테스트하지 않기로** 했으므로,
F1이 천장에 못 미쳐도 그게 (a) double-sparsity 때문인지 (b) 위치 재압축 때문인지 **분해 불가**. 한계로 명시.

### C.7 🟡 force_chunk_starts가 전역 sink(prefix[0])까지 강제 재계산
prefix[0]은 진짜 sink라 reuse가 옳은데 force_chunk_starts는 모든 청크 시작을 강제 → prefix[0]도 재계산
(무해하나 개념적 부정확 + 낭비). 또 force_last_chunk와 합쳐지면 forced 토큰이 많아져 "15% 예산" 해석이 흐려짐
(청크 많고 N=4면 forced가 budget을 압도). → force 대상을 doc 청크로 한정하거나 forced 비용을 측정·보고해야.

### C.8 🟡 sink=0(고립 압축)의 순환성
재사용성을 위해 sink=0(고립)으로 압축 → 그게 doc-sink 아티팩트를 **만들고** → 우리가 그걸 force-recompute로
**고친다**. 코히어런트하나, "스스로 만든 문제를 스스로 고치는" 구조라 순이득이 작을 수 있음(EPIC: 청크마다
sink N개는 과잉). 실측으로 순이득(sink 처리 on−off)이 양인지 확인 필요.

### C.9 🟡 KVzip 의존성/API 리스크
ModelKVzip API(prefill(do_score=True), kv.score/sink/ctx_len)가 KVzip 버전에 종속. 포팅이 그 API를 가정 →
버전 다르면 깨짐. no_sys_prompt(sink=0) 훅도 KVzip 내부에 의존. GPU 셋업에서 조기 sanity 필요.

### C.10 결론 (자기비판 요약)
이 계획은 **엔지니어링적으로는 깔끔**(기존 코드 재사용, 최소 추가)하나, **과학적으로는 두 개의 큰 미검증
가정**(C.1 HKVD가 압축 KV에서 회복한다, C.2 KVzip importance가 sink을 surface한다) 위에 서 있다. 둘 다
**과거 realistic-serving 역전처럼 데이터에서 뒤집힐 수 있다**. 따라서 다음 단계의 1순위는 "더 많은 knob"이
아니라 **C.1·C.2를 작은 실험으로 먼저 결판**내는 것이어야 한다. 그 전까지 sink ablation 그리드(S3)는 보류.
