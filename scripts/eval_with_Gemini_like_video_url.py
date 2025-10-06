import json
import os
import re
import json
import os
import re
from tqdm import tqdm

EVAL_MODEL = 'gemini-2.5-pro'

GITHUB_BASE_URL = '{BASEURL}/{}.mp4'
"""
Replace the BASEURL with your actual base URL where the videos are hosted.
"""

EVAL_PROMPT_FOR_OVERALL_EVENTS = (
    "These are several events that may happened in the video."
    "Please sort them in the order in which they occurred. "
    "If an event does not occur, the letter corresponding to the event is ignored."
    "Only output the corresponding letters in order, use commas to separate, and wrap your response with <output></output>. Such as <output>B,D</output> or <output>A,B,C</output>"
    "{}"
)

EVAL_PROMPT_FOR_STATISTIC_QUESTION = (
    "In this video."
    "\n{}\n"
    "Answer with Yes or No."
)

def request_api_video(prompt, model_name, video_path):
    """
    Replace this function with the actual API call to the model serving the video input.
    """
    
    return res

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
    
    eval_json_path = './prompt.json'
    eval_content = json.load(open(eval_json_path, "r"))
    output_root = './eval_res'
    output_path = os.path.join(output_root, 'eval_res.json')
    if os.path.exists(output_path):
        res_dict = json.load(open(output_path, "r"))
    else:
        res_dict = {}

    eval_keys = list(eval_content.keys())
    
    eval_number_mapping = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for eval_key in tqdm(eval_keys):
        if eval_key in res_dict:
            print('skip:', eval_key)
            continue
        try:
            video_path = GITHUB_BASE_URL.format(eval_key)

            if 't2v_eval_event_info' not in eval_content[eval_key]:
                continue
            verification_plan = eval_content[eval_key]['t2v_eval_event_info']['verification_plan']
            idx = 0
            overall_check = ''
            for v_plan in verification_plan:
                overall_check += '\n' + eval_number_mapping[idx] + ': ' + v_plan['event_description'] 
                idx += 1
            wrong_output_flag = True
            idx = 0
            while wrong_output_flag:
                idx += 1
                if idx==4:
                    break
                overall_prompt = EVAL_PROMPT_FOR_OVERALL_EVENTS.format(overall_check)
                output_response = request_api_video(
                    prompt=overall_prompt,
                    model_name=EVAL_MODEL,
                    video_path=video_path
                )
                print('res:', output_response, 'res type:', type(output_response))
                if '<output>' not in output_response or '</output>' not in output_response:
                    continue
                match = re.search(r'<output>(.*?)</output>', output_response, re.DOTALL)


                geted_output = match.group(1)

                res = geted_output


                eval_content[eval_key]['t2v_eval_event_info']['overall_event_res'] = res
                eval_content[eval_key]['t2v_eval_event_info']['overall_event_processed_res'] = filter_and_join_letters(res)
                wrong_output_flag = False

            verification_checks = eval_content[eval_key]['verification_checks']
            for v_check in verification_checks:
                if 'check' in v_check:
                    if v_check['check'] == False:
                        continue
                wrong_output_flag = True
                idx = 0
                while wrong_output_flag:
                    single_verification_check = EVAL_PROMPT_FOR_STATISTIC_QUESTION.format(v_check['question'])
                    
                    res = request_api_video(
                        prompt=single_verification_check,
                        model_name=EVAL_MODEL,
                        video_path=video_path
                    )

                    v_check['res'] = extract_first_yes_or_no(res)
                    idx += 1
                    if v_check['res'] != 'wrong' or idx==3:
                        wrong_output_flag=False
                

            res_dict[eval_key] = eval_content[eval_key]
            if len(list(res_dict.keys())) % 3 == 0:
                json.dump(res_dict, open(output_path, "w"))
        except Exception as e:
            print('Error:', e)

    json.dump(res_dict, open(output_path, "w"))
    pass