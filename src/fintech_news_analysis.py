"""Fintech news trend analysis (2016-2025).

This script consolidates the final analysis flow from the original Jupyter Notebook
HTML export used in the course project.

Expected input files:
    data/핀테크 뉴스 (2016-2018).csv
    data/핀테크 뉴스 (2019-2022).csv
    data/핀테크 뉴스 (2023-2025).csv
"""

from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
from konlpy.tag import Okt


DATA_DIR = Path("data")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

INPUT_FILES = [
    DATA_DIR / "핀테크 뉴스 (2016-2018).csv",
    DATA_DIR / "핀테크 뉴스 (2019-2022).csv",
    DATA_DIR / "핀테크 뉴스 (2023-2025).csv",
]

REQUIRED_COLUMNS = [
    "뉴스 식별자",
    "일자",
    "언론사",
    "기고자",
    "제목",
    "키워드",
    "특성추출(가중치순 상위 50개)",
    "본문",
    "URL",
    "분석제외 여부",
]


# ---------------------------------------------------------------------------
# 1. Rule-based relevance filtering
# ---------------------------------------------------------------------------

INCLUDE_KEYWORDS = [
    "핀테크", "간편결제", "삼성페이", "카카오페이", "토스", "인터넷전문은행",
    "카카오뱅크", "케이뱅크", "규제 샌드박스", "P2P 금융", "마이데이터",
    "오픈뱅킹", "블록체인", "가상자산", "암호화폐", "NFT", "메타버스 금융",
    "비대면 인증", "디지털 전환", "데이터 3법", "생성형 AI", "금융 AI",
    "챗봇", "금융 에이전트", "STO", "토큰 증권", "임베디드 금융", "슈퍼 앱",
    "트래블월렛", "조각 투자", "전자지갑", "간편송금", "디지털자산",
    "모바일결제", "비대면 금융", "로보어드바이저",
]

EXCLUDE_KEYWORDS = [
    "조류독감", "AI반도체", "반도체", "로봇", "자율주행", "게임", "엔터테인먼트",
    "야구", "축구", "부동산 분양", "배터리", "자동차", "의료 AI", "교육 AI",
    "국방", "군사", "농업", "바이오",
]

FINANCE_CONTEXT_WORDS = [
    "금융", "은행", "결제", "송금", "투자", "보험", "증권", "대출", "자산",
    "플랫폼", "서비스", "카드", "계좌", "인증", "보안", "신용", "테크",
    "데이터", "자산관리", "거래", "지갑",
]

EXCLUDE_KEYWORDS_2 = [
    "자동차", "전기차", "차체", "배터리", "부품", "제조사", "전장",
    "사물인터넷", "대통령 표창", "중견기업", "공장", "생산라인", "조선",
    "해운", "항공", "건설", "부동산", "의약", "의료기기", "게임", "엔터",
    "영화", "드라마", "야구", "축구",
]

CORE_FINTECH_KEYWORDS = [
    "오픈뱅킹", "마이데이터", "간편결제", "카카오페이", "삼성페이", "토스",
    "인터넷전문은행", "카카오뱅크", "케이뱅크", "블록체인", "디지털자산",
    "가상자산", "암호화폐", "NFT", "토큰증권", "증권형 토큰", "STO",
    "비대면 인증", "전자지갑", "간편송금", "로보어드바이저", "챗봇",
    "금융 AI", "생성형 AI", "슈퍼앱", "임베디드 금융", "조각 투자",
]


def count_keyword_matches(text, keyword_list):
    text = str(text).lower()
    return sum(1 for keyword in keyword_list if keyword.lower() in text)


def is_relevant_article(text):
    text = str(text)
    include_count = count_keyword_matches(text, INCLUDE_KEYWORDS)
    exclude_count = count_keyword_matches(text, EXCLUDE_KEYWORDS)
    finance_count = count_keyword_matches(text, FINANCE_CONTEXT_WORDS)

    if "핀테크" in text:
        return True
    if include_count >= 1 and finance_count >= 1:
        return True
    if include_count >= 2:
        return True
    if exclude_count >= 1 and finance_count == 0:
        return False
    return False


def is_relevant_article_v2(text):
    text = str(text)
    fintech_count = count_keyword_matches(text, CORE_FINTECH_KEYWORDS)
    exclude_count = count_keyword_matches(text, EXCLUDE_KEYWORDS_2)
    finance_count = count_keyword_matches(text, FINANCE_CONTEXT_WORDS)

    if fintech_count >= 1:
        return True
    if finance_count == 0 and exclude_count >= 1:
        return False
    if exclude_count >= 2 and fintech_count == 0:
        return False
    return True


# ---------------------------------------------------------------------------
# 2. Text cleaning and tokenization
# ---------------------------------------------------------------------------

STOPWORDS = {
    "있다", "했다", "한다", "위해", "통해", "이번", "지난", "관련", "대한", "있는",
    "하는", "에서", "으로", "까지", "에게", "및", "등", "더", "것", "수", "또", "한",
    "그", "이", "저", "때", "후", "전", "중", "내", "외", "가장", "정도", "경우",
    "모든", "각", "최근", "올해", "기자", "뉴스", "사진", "제공", "연합뉴스",
    "뉴시스", "머니투데이", "매일경제", "한국경제", "파이낸셜뉴스", "서울경제",
    "헤럴드경제", "말했다", "밝혔다", "설명했다", "전했다", "나타났다", "조사됐다",
    "기반", "추진", "확대", "강화", "활용", "도입", "제공", "운영", "출시",
    "서비스", "시장", "산업", "기업", "업계", "금융", "핀테크",
    "위한", "대해", "있으며", "있어", "없는", "한다고", "했다고", "체결했다고",
    "강조했다", "나섰다", "나서고", "보인다", "된다", "됐다", "내년", "이날",
    "당시", "우리", "양사", "각사", "것으로", "때문에", "때문이다", "가운데",
    "오전", "오후", "부터", "이후", "전면", "출처", "종합", "르포", "플랫폼",
    "국내", "글로벌", "전략", "체결", "협약", "공동", "업무협약", "파트너십",
    "금융위", "금융위원회", "금융권", "핀테크기업", "핀테크사",
}

EXTRA_STOPWORDS = {
    "테크", "한국", "대표", "업체", "지원", "업무", "서울", "위원회", "회사",
    "개발", "분야", "그룹", "전문", "고객", "사업", "서비스", "플랫폼", "국내",
    "글로벌", "최근", "올해", "지난해", "이날", "이번", "통해", "위한", "관련",
    "대한", "대해", "것", "것으로", "등", "및", "강화", "확대", "도입", "운영",
    "출시", "활용", "체결", "협약", "업무협약", "파트너십", "금융위", "금융위원회",
    "금융권",
}


def clean_text(text):
    text = str(text).replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^\)]*\)", " ", text)
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_with_okt(text, okt):
    tokens = []
    for word, pos in okt.pos(str(text), stem=True):
        if pos not in {"Noun", "Alpha"}:
            continue

        word = word.strip()
        if len(word) <= 1:
            continue
        if word.isdigit():
            continue
        if word in STOPWORDS:
            continue

        tokens.append(word)

    return tokens


def remove_extra_stopwords(token_list):
    return [token for token in token_list if token not in EXTRA_STOPWORDS]


MERGE_RULES = {
    ("블록", "체인"): "블록체인",
    ("오픈", "뱅킹"): "오픈뱅킹",
    ("마이", "데이터"): "마이데이터",
    ("인공", "지능"): "인공지능",
    ("암호", "화폐"): "암호화폐",
    ("가상", "자산"): "가상자산",
    ("토큰", "증권"): "토큰증권",
    ("디지털", "자산"): "디지털자산",
    ("간편", "결제"): "간편결제",
    ("간편", "송금"): "간편송금",
    ("비대면", "인증"): "비대면인증",
    ("전자", "지갑"): "전자지갑",
    ("자산", "관리"): "자산관리",
    ("신용", "평가"): "신용평가",
    ("대출", "관리"): "대출관리",
    ("이상", "거래"): "이상거래",
    ("카카오", "뱅크"): "카카오뱅크",
    ("카카오", "페이"): "카카오페이",
    ("네이버", "페이"): "네이버페이",
    ("케이", "뱅크"): "케이뱅크",
    ("토스", "뱅크"): "토스뱅크",
    ("트래블", "월렛"): "트래블월렛",
    ("생성", "AI"): "생성AI",
    ("금융", "AI"): "금융AI",
    ("슈퍼", "앱"): "슈퍼앱",
}


def merge_compound_tokens(token_list):
    merged_tokens = []
    i = 0

    while i < len(token_list):
        if i == len(token_list) - 1:
            merged_tokens.append(token_list[i])
            break

        pair = (token_list[i], token_list[i + 1])
        if pair in MERGE_RULES:
            merged_tokens.append(MERGE_RULES[pair])
            i += 2
        else:
            merged_tokens.append(token_list[i])
            i += 1

    return merged_tokens


# ---------------------------------------------------------------------------
# 3. Keyword categories and trend analysis
# ---------------------------------------------------------------------------

CORE_TECH_KEYWORDS = {
    "AI", "인공지능", "생성AI", "금융AI", "블록체인", "마이데이터", "오픈뱅킹",
    "토큰증권", "STO", "가상자산", "암호화폐", "NFT", "디지털자산",
    "비대면인증", "전자지갑", "신용평가",
}

BASE_TECH_KEYWORDS = {
    "디지털", "데이터", "모바일", "인터넷", "보안", "클라우드",
}

SERVICE_KEYWORDS = {
    "결제", "간편결제", "송금", "간편송금", "대출", "대출관리", "투자", "자산관리",
    "자산", "계좌", "보험", "증권", "카드", "거래", "거래소", "상품", "펀드",
    "코인", "해외", "관리", "인증", "청구", "납부",
}

COMPANY_KEYWORDS = {
    "카카오", "카카오페이", "카카오뱅크", "네이버", "네이버페이", "토스",
    "토스뱅크", "KB", "NH", "신한", "우리", "하나", "케이", "케이뱅크",
    "뱅크", "페이", "나무", "트래블월렛",
}

POLICY_KEYWORDS = {
    "규제", "완화", "샌드박스", "제도", "정책", "위원회", "금융위", "금융위원회", "당국",
}

INFRA_KEYWORDS = {"은행", "금융사", "금융기관", "금융회사"}


def classify_keyword(word):
    if word in CORE_TECH_KEYWORDS:
        return "핵심기술"
    if word in BASE_TECH_KEYWORDS:
        return "기반기술/패러다임"
    if word in SERVICE_KEYWORDS:
        return "서비스"
    if word in COMPANY_KEYWORDS:
        return "기업/플레이어"
    if word in POLICY_KEYWORDS:
        return "정책/규제"
    if word in INFRA_KEYWORDS:
        return "금융인프라"
    return "기타"


def get_top_words_by_year(data, year, top_n=15):
    year_tokens = []
    for token_list in data.loc[data["연도"] == year, "tokens_refined"]:
        year_tokens.extend(token_list)
    return Counter(year_tokens).most_common(top_n)


MAIN_CORE_TECH_KEYWORDS = [
    "AI", "인공지능", "블록체인", "마이데이터", "오픈뱅킹",
    "토큰증권", "STO", "가상자산", "NFT",
]


def build_top_keyword_table(data):
    rows = []
    for year in sorted(data["연도"].dropna().unique()):
        for rank, (word, count) in enumerate(
            get_top_words_by_year(data, year, top_n=15), start=1
        ):
            rows.append(
                {
                    "연도": int(year),
                    "순위": rank,
                    "키워드": word,
                    "빈도": count,
                    "범주": classify_keyword(word),
                }
            )
    return pd.DataFrame(rows)


def build_core_tech_trend(data):
    rows = []
    for year in sorted(data["연도"].dropna().unique()):
        year_df = data[data["연도"] == year]
        row = {"연도": int(year)}

        for keyword in MAIN_CORE_TECH_KEYWORDS:
            row[keyword] = int(
                year_df["tokens_refined"].apply(lambda tokens: keyword in tokens).sum()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_tech_service_cooccurrence(data):
    rows = []

    for year in sorted(data["연도"].dropna().unique()):
        year_df = data[data["연도"] == year]

        for tech in MAIN_CORE_TECH_KEYWORDS:
            related_services = []
            for tokens in year_df["tokens_refined"]:
                if tech in tokens:
                    related_services.extend(
                        token for token in tokens if token in SERVICE_KEYWORDS
                    )

            for rank, (service, count) in enumerate(
                Counter(related_services).most_common(10), start=1
            ):
                rows.append(
                    {
                        "연도": int(year),
                        "기술": tech,
                        "순위": rank,
                        "서비스": service,
                        "동시등장빈도": count,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. N-gram and phrase trends
# ---------------------------------------------------------------------------

SYNONYM_MAP = {
    "인공지능": "AI",
    "에이아이": "AI",
    "Artificial Intelligence": "AI",
    "artificial intelligence": "AI",
    "생성AI": "생성형AI",
    "생성형 AI": "생성형AI",
    "금융AI": "금융AI",
    "금융 AI": "금융AI",
}

NGRAM_STOPWORDS = {
    "은행", "투자", "결제", "증권", "대출", "카드", "상품", "진행", "참여",
    "개최", "최대", "센터", "정보", "구축", "성장", "경쟁", "해외", "코리아",
    "나무", "페이", "뱅크", "카카오", "네이버", "KB", "NH",
}

TARGET_PHRASES = {
    "생성형 AI": ["생성형 AI", "생성형AI", "생성 AI", "생성AI", "챗GPT", "ChatGPT", "GPT"],
    "금융 AI": ["금융 AI", "금융AI", "AI 금융", "AI금융"],
    "디지털 전환": ["디지털 전환", "디지털전환"],
    "디지털 자산": ["디지털 자산", "디지털자산"],
    "토큰 증권": ["토큰 증권", "토큰증권", "증권형 토큰", "STO"],
    "가상 자산": ["가상 자산", "가상자산"],
    "오픈 뱅킹": ["오픈 뱅킹", "오픈뱅킹"],
    "마이 데이터": ["마이 데이터", "마이데이터"],
    "간편 결제": ["간편 결제", "간편결제"],
    "자산 관리": ["자산 관리", "자산관리"],
    "신용 평가": ["신용 평가", "신용평가"],
    "비대면 인증": ["비대면 인증", "비대면인증"],
}

TECH_GROUPS = {
    "AI 계열": [
        "AI", "인공지능", "생성형 AI", "생성형AI", "생성AI",
        "금융AI", "금융 AI", "챗GPT", "GPT",
    ],
    "블록체인·디지털자산 계열": [
        "블록체인", "암호화폐", "가상자산", "가상 자산", "디지털자산",
        "디지털 자산", "NFT", "토큰증권", "토큰 증권", "STO",
    ],
    "데이터·개인금융 계열": [
        "마이데이터", "마이 데이터", "데이터", "자산관리",
        "자산 관리", "신용평가", "신용 평가",
    ],
    "금융 인프라 계열": [
        "오픈뱅킹", "오픈 뱅킹", "간편결제", "간편 결제", "간편송금",
        "간편 송금", "비대면인증", "비대면 인증", "전자지갑", "전자 지갑",
    ],
}


def normalize_synonyms(token_list):
    return [SYNONYM_MAP.get(token, token) for token in token_list]


def make_ngrams(token_list, n=2):
    tokens = [
        token for token in token_list
        if token not in NGRAM_STOPWORDS and len(token) > 1
    ]
    return [
        " ".join(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]


def build_ngram_table(data):
    rows = []

    for year in sorted(data["연도"].dropna().unique()):
        year_df = data[data["연도"] == year]

        bigrams = []
        trigrams = []

        for items in year_df["bigrams"]:
            bigrams.extend(items)
        for items in year_df["trigrams"]:
            trigrams.extend(items)

        for rank, (ngram, count) in enumerate(
            Counter(bigrams).most_common(20), start=1
        ):
            rows.append(
                {
                    "연도": int(year),
                    "유형": "bigram",
                    "순위": rank,
                    "표현": ngram,
                    "빈도": count,
                }
            )

        for rank, (ngram, count) in enumerate(
            Counter(trigrams).most_common(20), start=1
        ):
            rows.append(
                {
                    "연도": int(year),
                    "유형": "trigram",
                    "순위": rank,
                    "표현": ngram,
                    "빈도": count,
                }
            )

    return pd.DataFrame(rows)


def contains_any_phrase(text, phrase_list):
    text = str(text).lower()
    return any(phrase.lower() in text for phrase in phrase_list)


def build_phrase_trend(data):
    rows = []

    for year in sorted(data["연도"].dropna().unique()):
        year_df = data[data["연도"] == year]
        total_articles = len(year_df)

        for phrase_name, variants in TARGET_PHRASES.items():
            article_count = int(
                year_df["clean_text"].apply(
                    lambda text: contains_any_phrase(text, variants)
                ).sum()
            )

            rows.append(
                {
                    "연도": int(year),
                    "표현": phrase_name,
                    "기사수": article_count,
                    "연도전체기사수": total_articles,
                    "정규화빈도_1000건당": round(
                        article_count / total_articles * 1000, 2
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_tech_group_trend(data):
    rows = []

    for year in sorted(data["연도"].dropna().unique()):
        year_df = data[data["연도"] == year]
        total_articles = len(year_df)

        for group_name, keywords in TECH_GROUPS.items():
            article_count = int(
                year_df["clean_text"].apply(
                    lambda text: contains_any_phrase(text, keywords)
                ).sum()
            )

            rows.append(
                {
                    "연도": int(year),
                    "기술계열": group_name,
                    "기사수": article_count,
                    "연도전체기사수": total_articles,
                    "정규화빈도_1000건당": round(
                        article_count / total_articles * 1000, 2
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------

def load_data():
    missing_files = [path for path in INPUT_FILES if not path.exists()]
    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "다음 입력 파일을 찾을 수 없습니다:\n" + missing_text
        )

    frames = []
    for path in INPUT_FILES:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["source_file"] = path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def preprocess(news_df):
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in news_df.columns
    ]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")

    work_df = news_df[REQUIRED_COLUMNS + ["source_file"]].copy()

    text_cols = [
        "제목", "키워드", "특성추출(가중치순 상위 50개)",
        "본문", "언론사", "기고자", "URL",
    ]
    for column in text_cols:
        work_df[column] = work_df[column].fillna("").astype(str)

    work_df["분석제외 여부"] = (
        work_df["분석제외 여부"].fillna("").astype(str)
    )
    work_df["일자"] = pd.to_datetime(
        work_df["일자"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    work_df["연도"] = work_df["일자"].dt.year
    work_df["full_text"] = (
        work_df["제목"] + " " + work_df["본문"]
    ).str.strip()

    before_dedup = len(work_df)
    work_df = work_df.drop_duplicates(subset=["URL"], keep="first")
    work_df = work_df.drop_duplicates(
        subset=["제목", "본문"],
        keep="first",
    )
    after_dedup = len(work_df)

    work_df["is_relevant"] = work_df["full_text"].apply(
        is_relevant_article
    )
    filtered_df = work_df[work_df["is_relevant"]].copy()

    filtered_df["clean_text"] = filtered_df["full_text"].apply(clean_text)
    filtered_df["is_relevant_v2"] = filtered_df["full_text"].apply(
        is_relevant_article_v2
    )
    filtered_df = filtered_df[filtered_df["is_relevant_v2"]].copy()

    okt = Okt()
    filtered_df["tokens_okt"] = filtered_df["clean_text"].apply(
        lambda text: tokenize_with_okt(text, okt)
    )
    filtered_df["token_count_okt"] = filtered_df["tokens_okt"].apply(len)
    filtered_df = filtered_df[
        filtered_df["token_count_okt"] >= 3
    ].copy()

    filtered_df["tokens_cleaned"] = filtered_df["tokens_okt"].apply(
        remove_extra_stopwords
    )
    filtered_df["tokens_refined"] = filtered_df["tokens_cleaned"].apply(
        merge_compound_tokens
    )

    print("최초 기사 수:", len(news_df))
    print("중복 제거 후:", after_dedup)
    print("중복 제거 기사 수:", before_dedup - after_dedup)
    print("최종 분석 기사 수:", len(filtered_df))

    return filtered_df


def main():
    news_df = load_data()
    df = preprocess(news_df)

    top_keywords_df = build_top_keyword_table(df)
    tech_trend_df = build_core_tech_trend(df)
    cooccurrence_df = build_tech_service_cooccurrence(df)

    df["tokens_for_ngram"] = df["tokens_refined"].apply(
        normalize_synonyms
    )
    df["bigrams"] = df["tokens_for_ngram"].apply(
        lambda tokens: make_ngrams(tokens, n=2)
    )
    df["trigrams"] = df["tokens_for_ngram"].apply(
        lambda tokens: make_ngrams(tokens, n=3)
    )

    ngram_df = build_ngram_table(df)
    phrase_trend_df = build_phrase_trend(df)
    tech_group_trend_df = build_tech_group_trend(df)

    top_keywords_df.to_csv(
        RESULT_DIR / "연도별_상위키워드_범주보강버전.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tech_trend_df.to_csv(
        RESULT_DIR / "연도별_핵심기술키워드_기사수_보강버전.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cooccurrence_df.to_csv(
        RESULT_DIR / "기술_서비스_동시등장.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ngram_df.to_csv(
        RESULT_DIR / "시각화용_연도별_ngram_상위표현.csv",
        index=False,
        encoding="utf-8-sig",
    )
    phrase_trend_df.to_csv(
        RESULT_DIR / "시각화용_핵심표현_연도별추세.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tech_group_trend_df.to_csv(
        RESULT_DIR / "시각화용_기술계열_연도별추세.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n결과 저장 완료:", RESULT_DIR.resolve())
    print("\n연도별 핵심 기술 기사 수:")
    print(tech_trend_df.to_string(index=False))


if __name__ == "__main__":
    main()
