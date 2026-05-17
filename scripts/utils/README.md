# 유틸리티 스크립트

단계에 독립적인 범용 도구 모음.

## 스크립트

| 파일 | 설명 |
|------|------|
| `list_data.py` | 데이터 디렉토리 구조 탐색 및 파일 목록 출력 |

## list_data.py

데이터 폴더의 구조를 파악하고, 특정 확장자(wav, mp3, json)를 제외한 파일 목록을 출력합니다. `dataset_list_1.txt`, `dataset_list_2.txt`가 이 스크립트로 생성되었습니다.

### 사용법

```bash
# 기본 (data/ 폴더 스캔)
python scripts/utils/list_data.py

# 특정 경로 스캔
python scripts/utils/list_data.py D:\speechfake

# tree 형태 출력
python scripts/utils/list_data.py --tree

# 결과 파일로 저장
python scripts/utils/list_data.py D:\speechfake -o dataset_list_2.txt

# 제외할 확장자 지정
python scripts/utils/list_data.py --exclude wav mp3 flac json
```

### 옵션

| 옵션 | 설명 |
|------|------|
| `target` | 대상 디렉토리 (기본: repo/data) |
| `-t, --tree` | tree 형태 출력 |
| `--exclude EXT [EXT...]` | 제외할 확장자 (기본: wav mp3 json) |
| `-o, --output FILE` | 결과를 파일로 저장 |
