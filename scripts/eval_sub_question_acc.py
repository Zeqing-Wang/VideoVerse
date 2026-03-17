import json
from collections import defaultdict
from tqdm import tqdm

INPUT_FILE = "YOUR RESULT"
def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_sub_questions = 0
    yes_count = 0
    no_count = 0
    wrong_count = 0

    total_checks = 0
    passed_checks = 0

    total_videos = len(data)
    passed_videos = 0

    type_stats = defaultdict(lambda: {
        "sub_total": 0,
        "sub_yes": 0,
        "sub_no": 0,
        "checks_total": 0,
        "checks_pass": 0
    })

    for video_id in tqdm(data):

        video_pass_flag = True
        verification_checks = data[video_id].get("verification_checks", [])

        for check in verification_checks:

            if "sub_question_results" not in check:
                continue

            check_type = check.get("check_type", "Unknown")

            total_checks += 1
            type_stats[check_type]["checks_total"] += 1

            sub_results = check["sub_question_results"]

            check_pass_flag = True

            for item in sub_results:

                total_sub_questions += 1
                type_stats[check_type]["sub_total"] += 1

                if item["res"] == "yes":
                    yes_count += 1
                    type_stats[check_type]["sub_yes"] += 1
                elif item["res"] == "no":
                    no_count += 1
                    type_stats[check_type]["sub_no"] += 1
                    check_pass_flag = False
                else:
                    wrong_count += 1
                    check_pass_flag = False

            if check_pass_flag:
                passed_checks += 1
                type_stats[check_type]["checks_pass"] += 1
            else:
                video_pass_flag = False

        if video_pass_flag:
            passed_videos += 1

    print("=" * 50)
    print("Overall Sub-question Accuracy:")
    if (yes_count + no_count) > 0:
        print("Sub-question Accuracy:",
              yes_count / (yes_count + no_count))
    print("Total Sub-questions:", total_sub_questions)
    print("Yes:", yes_count)
    print("No:", no_count)
    print("Wrong:", wrong_count)

    print("=" * 50)
    print("Check-level Accuracy:",
          passed_checks / total_checks if total_checks > 0 else 0)
    print("Total Checks:", total_checks)
    print("Passed Checks:", passed_checks)

    print("=" * 50)
    print("Video-level Accuracy:",
          passed_videos / total_videos if total_videos > 0 else 0)
    print("Total Videos:", total_videos)
    print("Passed Videos:", passed_videos)

    print("=" * 50)
    print("Per Check-Type Statistics:")
    for check_type, stats in type_stats.items():

        sub_acc = (
            stats["sub_yes"] / (stats["sub_yes"] + stats["sub_no"])
            if (stats["sub_yes"] + stats["sub_no"]) > 0 else 0
        )

        check_acc = (
            stats["checks_pass"] / stats["checks_total"]
            if stats["checks_total"] > 0 else 0
        )

        print(f"\n[{check_type}]")
        print("Sub-question Accuracy:", sub_acc)
        print("Check-level Accuracy:", check_acc)
        print("Total Checks:", stats["checks_total"])
        print("Total Sub-questions:", stats["sub_total"])

    print("=" * 50)


if __name__ == "__main__":
    main()