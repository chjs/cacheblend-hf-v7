# cacheblend-hf-v7 코드 리뷰 & 수정 체크리스트 (2026-06-09)

논문 **CacheBlend (EuroSys '25, arXiv:2405.16444)** 와 대조한 비판적 리뷰.
이 repo는 CacheBlend 알고리즘의 **quality-fidelity 재구현**(HF eager/SDPA)이다.
알고리즘부(pre-RoPE 저장·재-RoPE, HKVD, sparse forward)는 정확하나, **논문의 시스템
기여(pipelining·loading controller·TTFT/throughput)는 미구현/死코드**이고, 베이스라인
오염·버전 skew·하네스 가정 결함이 있다.

심각도: 🔴 Critical / 🟠 High / 🟡 Medium / 🟢 Low. 각 항목에 위치·문제·수정·검증 포함.

> **수정 진행 (compblend 브랜치):** C2 ✅ / H1 ✅ / H2 ✅ / H3 ✅ (아래). 회귀 테스트 tests/
> (4.51.3 venv): C2 3 + H1 2 + H2 3 + H3 4 + 하드닝 2 = 14 checks PASS. 어드버서리얼 재감사 결과는
> 맨 아래 "## 어드버서리얼 재감사" 절 참조 (N1/N2/N3 수정 반영).

---

## 🔴 Critical

### [ ] C1. 시스템 기여 미구현 — 그런데 TTFT를 측정·보고함
- **위치**: `fusor.py:262–272`(동기 일괄 로딩), `kv_store.py:99`(`prefetch_chunk` 死코드),
  `controller.py` 전체(임의 상수, 미호출), `benchmarks/musique/blend_musique_generic.py:204–223`,
  `runners.py`(ttft 측정).
- **문제**: pipelining 없음(K/V를 forward 전 전부 동기 로딩). `prefetch_chunk`/`LoadingController`는
  아무도 호출 안 함. 그럼에도 ttft를 측정·출력 → 순수 PyTorch eager/SDPA + 매 layer 전체길이
  텐서 재조립(P1) 때문에 이 TTFT는 **논문 주장을 입증 불가**, 작은 입력에선 full recompute보다
  느릴 수도.
- **수정(택1)**:
  1. (권장) TTFT 보고 제거 + README/리포트에 "이 구현은 **quality-only**, latency 측정 대상 아님" 명시.
  2. controller/prefetch를 死코드로 두지 말고 삭제하거나 `experimental/`로 격리.
- **검증**: benchmark 출력에서 ttft 컬럼 제거 또는 "(not a latency benchmark)" 라벨 확인.

### [ ] C2. transformers 버전 skew — `past_key_value`(단수) kwarg가 조용히 무시될 위험
- **위치**: `requirements.txt:16`(`transformers==4.49.0`), `fusor.py:313–322`, `model.py:153–161`.
- **문제**: 코드는 HF 4.49 가정 → `past_key_value=`(단수)로 디코더 layer 호출. 실제 실행 환경은
  KVzip이 강제하는 **4.51.3**(메모: dependency-stack-constraints). 4.51에서 kwarg가 바뀌었다면
  단수 인자가 `**kwargs`로 흘러 **조용히 무시 → DynamicCache가 비어** layers 0..check_layer-1의
  KV 미적재 → decode 붕괴. 가드 없음.
- **수정**:
  1. layers 0..check_layer-1 직후 가드 추가:
     `assert past_key_values.get_seq_length() > 0, "cache empty — past_key_value kwarg ignored?"`
     (또는 `len(past_key_values) == check_layer` 확인).
  2. `requirements.txt`를 실제 환경(`transformers==4.51.3`)과 일치.
  3. 단수/복수 kwarg를 버전 분기 처리하거나, 표준 `layer(...)` 대신 명시적 attention 호출.
- **검증**: 4.51.3 환경에서 N=2 smoke 후 `past_key_values.get_seq_length() == total_seq` 확인.

---

## 🟠 High

### [x] H1. FullReuse 베이스라인의 prefill과 decode가 다른 KV를 씀 (실험 오염) — ✅ FIXED
> **수정 완료** (compblend 브랜치): `FullReuseRunner`·`PrefixCacheRunner`가 두 번째 hook-less
> full forward를 버리고 `fuse_full_reuse`/`fuse_prefix_cache`의 자체 캐시(`return_layerwise_output=True`)로
> decode. tests/test_h1_reuse_cache.py로 검증 — 재사용 캐시 적재 확인 + 재사용 캐시 ≠ full-recompute
> 캐시(마지막 layer K 최대차 0.70)로 구버그 입증. transformers 4.51.3에서 C2+H1 전부 PASS.
- **위치**: `runners.py:231–243`.
- **문제**: `fuse_full_reuse`로 prefill logits를 얻은 뒤, decode용 `past_key_values`를 얻으려고
  **훅 없는 plain full forward를 한 번 더** 실행 → decode 캐시가 full-recompute KV가 됨. FullReuse가
  "첫 토큰만 reuse, 이후는 full-recompute"인 잡종 베이스라인이 되고 계산도 2×.
- **수정**: `fuse_full_reuse`가 `return_layerwise_output=True`로 **자기 past_key_values를 반환**하도록
  하고(이미 시그니처 존재), 그걸 decode에 그대로 사용. 두 번째 full forward 삭제.
  (CacheBlendRunner는 이미 이 방식 — 동일 패턴으로 통일.)
- **검증**: FullReuse decode가 reuse KV를 쓰는지 — 두 번째 `self.model(...)` 호출이 사라졌는지 확인.

### [x] H2. BOS / 토큰 경계 불일치 (runners 한정) — ✅ FIXED
> **수정 완료** (compblend 브랜치): `chunk_texts(..., prepend_bos=True)` 추가(BOS를 chunk 0에만),
> runners의 `_build_chunks`가 이를 사용, `FullRecomputeRunner`도 동일 `fused_input_ids(chunks)`를
> 소비(전체 문자열 재토크나이즈 제거). 이제 4개 runner가 동일 토큰열 사용. tests/test_h2_tokenization.py
> 검증 — 단일 토큰열 공유(len=61), BOS는 chunk0만, 구 경로(58)≠새 경로(61)로 구버그 입증.
- **위치**: `runners.py:175–181`(`FullRecomputeRunner`, `add_special_tokens` 기본 True → BOS 포함)
  vs `chunker.py:48`(`chunk_texts`, `add_special_tokens=False` → BOS 없음 + 청크별 토크나이즈 후 concat).
- **문제**: baseline과 CacheBlend가 **서로 다른 토큰 시퀀스/길이**(BOS 유무 + 경계 토큰화 차이)로
  평가됨. BOS는 attention-sink로 작동 → 품질에 비자명한 영향.
- **참고**: 핵심 benchmark `blend_musique_generic.py`는 양쪽 다 `fused_input_ids`(BOS 없음)라
  **이 결함에서 자유로움** → 결함은 runners.py에 국한. runners 경로를 평가에 쓸 때만 수정 필요.
- **수정**: `FullRecomputeRunner`도 `fused_input_ids(chunks)` 경로로 통일하거나, 양쪽 BOS 정책을
  일치(둘 다 추가 또는 둘 다 미추가)시킬 것.
- **검증**: 동일 예제에서 baseline/CacheBlend의 입력 토큰 길이가 일치하는지 print.

### [x] H3. 질의가 "재사용 가능한 압축 청크"로 취급됨 (realistic-serving 역전의 근원) — ✅ FIXED (opt-in)
> **수정 완료** (compblend 브랜치): `fuse_selective`에 `force_last_chunk: bool=False` 추가. True+다중청크면
> 마지막 청크(질의) 전체를 강제 fresh, `recompute_ratio`는 문서에만 적용(질의 토큰 별도 카운트):
> `recompute_k = n_forced + int((total−n_forced)*ratio)`. compblend7 `fuse_selective_compblend`의
> forced/masked-topk 의미 1:1. `hkvd.select_top_k_masked` 추가, `CacheBlendRunner.force_last_chunk` +
> benchmark `CACHEBLEND_FORCE_LAST_CHUNK` env 연결. **기본 False → 레거시 동작 bit-identical**(테스트 A로
> 증명). 추가 교정: `ratio==0 → full_reuse` shortcut을 `not force_last_chunk`로 게이팅(질의는 ratio=0에서도
> fresh; compblend7의 잠재 코너 수정, ratio>0 결과 불변). tests/test_h3_force_last_chunk.py 4건:
> A(레거시 동일), B(질의 전체+문서예산 분리), C(e2e True=6/6 vs False=1/6 질의 재계산), D(ratio0 게이팅).
- **위치**: `benchmarks/musique/blend_musique_generic.py:189–199`, `runners.py:304–307`,
  force-include 마지막 위치만 fresh `fusor.py:344–352`.
- **문제**: 질의 청크 `[q_prompt+assistant_open]`를 precompute해 kv_store에 넣고 `fuse_selective`가
  그 안에서 HKVD만 재계산. 메모(realistic-serving-reversal)가 "+0.016★는 query-budget 아티팩트"로
  규명한 바로 그 설정. **realistic-serving 수정(질의 전체를 항상 full-prefill)은 이 repo에 없음**
  (compblend7에만 존재).
- **수정**: `fuse_selective`에 `force_last_chunk`(또는 `query_chunk_idx`) 옵션을 추가해 질의 청크를
  reuse 대상에서 제외하고 항상 full-prefill. 최소한 README/주석에 "이 하네스는 질의를 재사용
  청크로 취급 — 결과 해석 주의" 명시.
- **검증**: 질의 청크의 모든 토큰이 top_indices에 포함(=fresh)되는지 확인.

---

## 🟡 Medium

### [ ] P1. 매 layer 전체길이 cached-K RoPE 재조립 — 비효율 + 메모리 스파이크
- **위치**: `fusor.py:439–451`(매 layer `apply_rotary_pos_emb(zeros_like(K), K_stored_pre[li], ...)`
  + `.clone()` + scatter), `:464–465`(V `repeat_interleave`로 full num_heads 확장).
- **문제**: sparse Q가 줄이는 건 proj/FFN뿐, K/V 준비는 O(S·hidden) full-length. Loong(50k+)에서
  layer마다 큰 transient. `dummy_q=zeros_like` 후 폐기도 낭비.
- **수정**: (1) dummy_q 없이 K만 회전하는 헬퍼로 교체. (2) PyTorch ≥2.5면 SDPA `enable_gqa=True`로
  `repeat_interleave` 제거. (3) cached-K RoPE 결과를 가능하면 캐시(단, layer별로 달라 제한적).
- **검증**: Loong 1예제 peak memory / layer-time 비교.

### [ ] P2. 단일 check_layer deviation — 논문의 gradual filtering 미구현
- **위치**: `fusor.py:339–340`(check_layer 한 층 deviation으로 HKVD 확정).
- **문제**: 논문 §4.3의 layer별 점진 필터링(r₁>r₂>…) 미구현. LMCache 1:1로는 일관되나 **논문
  충실도 간극**. 메모(design-sweep-results: check_layer=8이 큰 레버)와 직결.
- **수정(택1)**: (a) 리포트에 "단일 check_layer 근사, gradual filtering 미구현" 명시(저비용).
  (b) gradual filtering 구현 — layer별 후보 집합을 좁히며 deviation 재계산.
- **검증**: check_layer 스윕에서 품질 변화 재현.

### [ ] P3. `logits_full`이 top 위치 외 전부 0 (footgun)
- **위치**: `fusor.py:490–491`.
- **문제**: greedy decode는 마지막 위치만 써서 정상이나, perplexity/teacher-forcing/다중 위치
  scoring에 재사용 시 조용히 틀림.
- **수정**: docstring 넘어 런타임 가드(예: 마지막 위치만 반환하는 경로 분리) 또는 반환 객체에
  `valid_positions=top_indices` 메타 부착.
- **검증**: scoring 용도 호출처가 없는지 grep, 있으면 가드.

### [ ] P4. 고립 prefill의 sink 아티팩트
- **위치**: `precompute.py:precompute_chunk_kv`(각 청크 BOS/prefix/sink 없이 0..L-1 단독 prefill).
- **문제**: 메모(loong-isolated-prefill-sink) — 각 청크 첫 토큰이 비정상 sink K → HKVD/deviation에
  잡음. `precompute_from_cache_prompt`(cross-chunk 버전) 있으나 benchmark/runner는 단독 버전 사용.
- **수정**: 설계상 trade-off. 분석 시 통제 변수로 두거나, sink 토큰 prepend 옵션 추가 후 영향 측정.
- **검증**: sink 유무에 따른 deviation 분포 변화 비교.

---

## 🟢 Low / 관찰

- [ ] **L1.** `model.py:239` `__del__` 훅 제거는 GC 타이밍 의존. `runners.py:204`
  `LayerwiseModel.__new__` 우회 생성이 같은 모델에 k_proj 훅을 **중복 설치**할 수 있음 → 훅
  누수/중복 캡처 점검.
- [ ] **L2.** `hkvd.py:50` `kv_deviation` batch>1 미지원(명시적 예외) → 배치 평가 불가.
- [ ] **L3.** `tolerance.py:63` `IDENTICAL_PATH`는 `max_diff==0.0` — boundary shortcut 외엔
  FP16 비결정성으로 통과 어려움(의도된 엄격함이나 fragile).
- [ ] **L4.** `controller.py` 상수(1/4/12/50, 1.0~3.3)는 "illustrative" 명시 — paper 수치(RAM↔SSD
  ~10×, r*=15%)와 연결 안 됨, 논문 재현엔 못 씀(C1과 함께 처리).

---

## 정확하다고 확인된 부분 (수정 불필요)
- pre-RoPE 저장 + 전역 위치 재-RoPE: 직교회전 → squared-L2 보존 → pre/post-RoPE deviation
  동일(`fusor.py:337–339` 주석의 수학 주장 맞음).
- GQA `repeat_interleave(n_rep, dim=1)` == HF `repeat_kv`.
- HKVD 선택(`hkvd.py`)은 LMCache `blender.py:89–101`과 1:1.
- DynamicCache layer 순차 update, kv-head 수로 저장 — decode 경로와 일관.

---

## 권장 수정 순서
1. **C2 가드** (즉시, 저비용, 조용한 붕괴 차단)
2. **H1 / H2** (베이스라인 공정성 — 평가 신뢰도 직결)
3. **H3 옵션 추가** (realistic-serving 정합 — 연구 결론과 직결)
4. **C1 라벨링** (TTFT 보고 정리 / 死코드 격리)
5. P1–P4, L1–L4 (여력 시)

## 연구적 시사점
H3 + P2 + P4가 결합해 **"deviation 신호의 품질"이 이미 하네스 안에서 왜곡**되어 있었고, 이는
메모리의 realistic-serving 역전(importance가 HKVD를 못 이김 / Gated-HKVD 미재현)을 **코드 레벨에서
일관되게 설명**한다. 수정 시 이 인과를 깨지 않도록 H3를 우선 정합할 것.

---

## 어드버서리얼 재감사 (2026-06-09, C2/H1/H2/H3 수정 후)

수정이 의도대로인지 + 새 버그 유발 여부를 독립 리뷰어로 재검증. 신규 문제 3건 발견·수정.

### [x] N1 (Medium) — 빈 청크 → 0-length forward 크래시 + C2 가드 무의미 통과 ✅ FIXED
- token_ids=[] 청크는 `precompute_chunk_kv`에서 (1,0) forward → HF 내부 크래시. 게다가 C2 가드
  `assert got==seq`가 seq=0이면 0==0으로 통과(무의미). 실무상 musique는 "Document N:" 래퍼로 빈
  청크가 안 생기지만 라이브러리 레벨 결함.
- 수정: `precompute_chunk_kv`에 0-token 청크 거부 가드(ValueError). test_hardening.py N1.

### [x] N2 (Medium) — C2 수정의 잔여 위험: cache_name=None 시 cache 조용한 누락 ✅ FIXED
- `_layer_spec`이 래핑/소거된 forward 시그니처에서 cache 파라미터를 못 찾으면 cache_name=None →
  call_decoder_layer가 cache를 누락(=C2가 막으려던 그 실패). 현재 핀(4.51–4.52)에선 안 터지나
  "절대 안 떨어진다"는 주장은 그 분기에서 거짓.
- 수정: cache_name=None이고 past_key_values 있으면 즉시 RuntimeError(loud-fail). test_hardening.py N2.

### [x] N3 (Low) — force 경로 recompute_k에 max(.,1) 부재 ✅ FIXED
- `recompute_k = n_forced + int(...)`에 명시적 하한 없음(현재는 n_forced≥1 + 하류 clamp로 안전).
  수정: 두 분기 뒤 `recompute_k = max(int(recompute_k), 1)`로 통일.

### 🟢 N4 (Low, 관찰만) — 디코드 중 persistent pre-RoPE 캡처 훅 활성
- `LayerwiseModel._install_k_proj_hooks`의 캡처 훅이 decode 스텝 동안에도 작동(`_pre_rope_k`에 기록).
  반환 안 되므로 무해하나 낭비 + foot-gun. `__new__`로 만든 shim 모델은 `__del__` 미보장으로 훅 미제거.

### 검증됨(의심스러웠으나 정확) — 독립 리뷰어가 2000회 랜덤+타이 케이스로 확인
- select_top_k_masked는 force_last_chunk=False에서 레거시와 **bit-identical**(타이브레이크·집합크기 포함).
- forced 위치는 +inf·target_k=max(recompute_k,n_forced)로 **항상 포함 보장**(forced가 98/100여도).
- force_last_chunk=True가 질의 청크를 **전 layer fresh**로 처리; 큰/연속 top_indices에서도 sparse
  forward(인덱싱·마스크·cos_sparse)가 정확. top_indices가 작다는 가정 없음.
- ratio==0+force_last_chunk 게이팅(D-case): 질의만 fresh, 문서 전부 재사용 — 정확.
- C2 가드는 sliding-window/GQA/batch=1에서 false-positive 없음. H2 chunk_id 안정·double-BOS 무위험
  (Mistral/Llama/Qwen은 add_special_tokens=False면 BOS 미부착), chunk_texts 다른 호출자 무영향.
- 에이전트 과장 정정: "check_layer==0이 fuse_selective C2 가드 우회"는 실홀 아님 — check_layer==0이면
  call_decoder_layer 루프(range(0))가 안 돌아 누락할 cache 자체가 없음. forward_layerwise 가드는 항상 작동.
