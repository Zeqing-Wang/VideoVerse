import json
import os
from collections import defaultdict

def calculate_lcs(s1: str, s2: str) -> str: 
    n, m = len(s1), len(s2)
    dp = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + s1[i - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], key=len)

    return dp[n][m]


def analyze_test_data(data: dict) -> dict:
    report = {
        "overall_summary": {},
        "per_video_analysis": {}
    }
    all_verification_checks = []
    total_event_lcs_length = 0

    for video_id, video_data in data.items():
        video_report = {}
        event_info = video_data.get("t2v_eval_event_info", {})
        verification_plan = event_info.get("verification_plan", [])
        
        if verification_plan:
            num_events = len(verification_plan)
            ground_truth_string = "".join([chr(ord('A') + i) for i in range(num_events)])
            
            predicted_string = event_info.get("overall_event_processed_res", "")
            
            len_lcs = len(calculate_lcs(ground_truth_string, predicted_string))
            video_report["event_lcs_length"] = len_lcs
            total_event_lcs_length += len_lcs
        else:
            video_report["event_lcs_length"] = None 
        

        checks = video_data.get("verification_checks", [])
        if checks:
            scores_by_category = defaultdict(list)
            total_scores = []

            for check in checks:
                score = 1 if check.get("res", "").lower() == "yes" else 0
                category = check.get("check_type", "uncategorized")
                
                scores_by_category[category].append(score)
                total_scores.append(score)
                all_verification_checks.append({"category": category, "score": score})

            category_scores = {
                cat: sum(scores) for cat, scores in scores_by_category.items()
            }
            overall_score = sum(total_scores)

            video_report["verification_checks_analysis"] = {
                "overall_score": overall_score,
                "scores_by_category": category_scores,
                "total_checks": len(total_scores)
            }
        else:
            video_report["verification_checks_analysis"] = None

        report["per_video_analysis"][video_id] = video_report

    num_videos = len(data)
    if num_videos > 0:
        report["overall_summary"]["total_event_adherence"] = total_event_lcs_length

        if all_verification_checks:

            report["overall_summary"]["verification_checks_score"] = sum(check['score'] for check in all_verification_checks)

            overall_scores_by_cat = defaultdict(list)
            for check in all_verification_checks:
                overall_scores_by_cat[check['category']].append(check['score'])
            category_accumulative_score = {
                cat: sum(scores) for cat, scores in overall_scores_by_cat.items()
            }
            report["overall_summary"]["verification_checks_score_by_category"] = category_accumulative_score
        
        report["overall_summary"]["total_score"] = total_event_lcs_length + report["overall_summary"].get("verification_checks_score", 0)
        report["overall_summary"]["total_videos_processed"] = num_videos

    return report


def process_file(input_path: str):

    with open(input_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)


    final_report = analyze_test_data(test_data)

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_report.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    eval_res_path = 'REPLACE_WITH_YOUR_PATH'
    process_file(eval_res_path)