# Smith-Waterman 알고리즘 Python 구현 (논문 기반 + 주석 상세 정리)
# 논문: Smith & Waterman (1981) - "Identification of Common Molecular Subsequences"
# 목표: 두 생물학적 시퀀스(seq1, seq2)에서 가장 유사한 부분서열 쌍을 찾기
# 특징: 국소 정렬(Local Alignment) 알고리즘, 삽입/삭제(gap), mismatch, match 고려

import numpy as np

def smith_waterman(seq1, seq2, match=2, mismatch=-1, gap_open=-2):
    """
    Smith-Waterman 알고리즘 구현 함수
    - seq1, seq2: 정렬할 두 시퀀스 (문자열)
    - match: 문자 일치 시 점수 (default=+2)
    - mismatch: 문자 불일치 시 벌점 (default=-1)
    - gap_open: 길이 1짜리 공백 삽입 시 벌점 (default=-2)

    반환값:
    - aligned1: 정렬된 seq1 부분 문자열
    - aligned2: 정렬된 seq2 부분 문자열
    - max_score: 최고 유사도 점수
    - H: 유사도 행렬 전체
    """

    n, m = len(seq1), len(seq2)  # 논문: A = a1...an, B = b1...bm

    H = np.zeros((n+1, m+1), dtype=int)  # H[i][j] = a_i와 b_j까지 고려한 최대 유사도
    traceback = np.zeros((n+1, m+1), dtype=int)  # 경로 복원용: 0=끝, 1=↖, 2=↑, 3=←

    max_score = 0
    max_pos = (0, 0)  # 최고점 좌표 저장

    # H 행렬 채우기 (논문 수식 (1)과 대응)
    for i in range(1, n+1):
        for j in range(1, m+1):
            # 유사도 계산 s(ai, bj)
            if seq1[i-1] == seq2[j-1]:
                score = match
            else:
                score = mismatch

            # 각 방향에 대한 점수 계산
            diag = H[i-1][j-1] + score  # 대각선: match/mismatch
            up = H[i-1][j] + gap_open   # 위쪽: seq2에 갭 (seq1 문자 유지)
            left = H[i][j-1] + gap_open # 왼쪽: seq1에 갭 (seq2 문자 유지)

            # 국소 정렬 → 음수 점수 리셋
            H[i][j] = max(0, diag, up, left)

            # traceback 방향 설정
            if H[i][j] == 0:
                traceback[i][j] = 0
            elif H[i][j] == diag:
                traceback[i][j] = 1  # ↖
            elif H[i][j] == up:
                traceback[i][j] = 2  # ↑
            else:
                traceback[i][j] = 3  # ←

            # 최대 점수 위치 저장
            if H[i][j] >= max_score:
                max_score = H[i][j]
                max_pos = (i, j)

    # traceback으로 최적 부분서열 복원
    aligned1 = ""
    aligned2 = ""
    i, j = max_pos

    while traceback[i][j] != 0:
        if traceback[i][j] == 1:
            aligned1 = seq1[i-1] + aligned1
            aligned2 = seq2[j-1] + aligned2
            i -= 1
            j -= 1
        elif traceback[i][j] == 2:
            aligned1 = seq1[i-1] + aligned1
            aligned2 = '-' + aligned2
            i -= 1
        elif traceback[i][j] == 3:
            aligned1 = '-' + aligned1
            aligned2 = seq2[j-1] + aligned2
            j -= 1

    return aligned1, aligned2, max_score, H

# 🎯 보충 설명: 논문 수식의 k란?
# 논문 수식 (1)에서 나오는 k는 "갭의 길이"를 의미
#   H[i-k][j] - W_k  → 위쪽으로 k칸 건너뛰며 공백 삽입 (seq2 기준 갭)
#   H[i][j-l] - W_l  → 왼쪽으로 l칸 건너뛰며 공백 삽입 (seq1 기준 갭)
#
# 이 구현은 단순화를 위해 k=1만 고려함 (즉, gap 길이 1에 대해 고정 패널티 적용)
# 논문처럼 여러 k를 고려하려면 아래처럼 max를 반복해야 함:
#   up = max([H[i-k][j] - gap_penalty(k) for k in range(1, i+1)])
#   left = max([H[i][j-k] - gap_penalty(k) for k in range(1, j+1)])

# 예시 실행 (필요 시 주석 해제)
if __name__ == "__main__":
    seqA = "AUGCCAUUGACGG"
    seqB = "CAGCCUCGCUUAG"

    a1, a2, score, matrix = smith_waterman(seqA, seqB)
    print("정렬 결과:")
    print(a1)
    print(a2)
    print("최대 유사도 점수:", score)
    print("H 행렬:")
    print(matrix)
