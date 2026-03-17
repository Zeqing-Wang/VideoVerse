
import json
import os
import re
import base64
import json
import mimetypes
import os
import requests
import re
from request_api_zoo import request_api_openai_video
from tqdm import tqdm
EVAL_MODEL = 'gemini-2.5-pro'

GITHUB_BASE_URL = '{BASEURL}/{}.mp4'
EVAL_PROMPT_FOR_OVERALL_EVENTS = (
    "These are several events that may happened in the video."
    "Please sort them in the order in which they occurred. "
    "If an event does not occur, the letter corresponding to the event is ignored."
    "Only output the corresponding letters in order, use commas to separate, and wrap your response with <output></output>. Such as <output>B,D</output> or <output>A,B,C</output>"
    "{}"
)
EVAL_PROMPT_FOR_SINGLE_EVENT_EXISTING = (
    "This is a Generated Video. Does the following event existing in this video?"
    "\n{}\n"
    "Answer with Yes or No."
)
EVAL_PROMPT_FOR_STATISTIC_QUESTION = (
    "In this video."
    "\n{}\n"
    "Answer with Yes or No."
)

def filter_and_join_letters(input_string: str) -> str:

  filtered_chars = [char for char in input_string if 'A' <= char <= 'Z']
  

  return "".join(filtered_chars)


def extract_first_yes_or_no(text: str):
  
  match = re.search(r'yes|no', text, re.IGNORECASE)

  if match:

    return match.group(0).lower()
  else:
    return 'wrong'

if __name__ == '__main__':

    eval_json_path = 'prompt/prompts_of_VideoVerse_decomposed.json'


    eval_content = json.load(open(eval_json_path, "r"))

    output_root = './eval_res'
    output_dir = os.path.join(output_root, model_name)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_path = os.path.join(output_dir, 'eval_res_sub_questions.json')

    if os.path.exists(output_path):
        print('^'*15)
        print('Resume from:', output_path)
        print('^'*15)
        res_dict = json.load(open(output_path, "r"))
    else:
        res_dict = {}

    eval_keys = list(eval_content.keys())

    for eval_key in tqdm(eval_keys):

        if eval_key in res_dict:
            print('skip:', eval_key)
            continue

        try:
            video_path = GITHUB_BASE_URL.format(model_name, eval_key)


            verification_checks = eval_content[eval_key]['verification_checks']

            for v_check in verification_checks:


                if 'decomposed_sub_questions' not in v_check:
                    continue

                sub_questions = v_check['decomposed_sub_questions']


                v_check['sub_question_results'] = []

                for sub_q in sub_questions:

                    wrong_output_flag = True
                    retry_idx = 0

                    while wrong_output_flag:

                        single_prompt = EVAL_PROMPT_FOR_STATISTIC_QUESTION.format(sub_q)

                        print('+'*10)
                        print('Sub Question:', single_prompt)
                        print('Video Path:', video_path)
                        print('-'*10)

                        res = request_api_openai_video(
                            prompt=single_prompt,
                            model_name=EVAL_MODEL,
                            video_path=video_path
                        )

                        parsed_res = extract_first_yes_or_no(res)

                        retry_idx += 1

                        if parsed_res != 'wrong' or retry_idx == 3:
                            wrong_output_flag = False

                    v_check['sub_question_results'].append({
                        "question": sub_q,
                        "res": parsed_res
                    })


                all_yes = all(
                    item['res'] == 'yes'
                    for item in v_check['sub_question_results']
                )

                v_check['all_sub_questions_yes'] = all_yes

            res_dict[eval_key] = eval_content[eval_key]

            if len(res_dict) % 3 == 0:
                json.dump(res_dict, open(output_path, "w"))

        except Exception as e:
            print('Error:', e)

    json.dump(res_dict, open(output_path, "w"))
