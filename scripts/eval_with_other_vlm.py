# Dummyfrom transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import json
import os
import re
import base64
import json
import mimetypes
import os
import requests
import re
# from request_api_zoo_video import request_api_openai_video
from tqdm import tqdm

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu    # pip install decord
from scipy.spatial import cKDTree
import numpy as np
import math
MAX_NUM_FRAMES=180 # Indicates the maximum number of frames received after the videos are packed. The actual maximum number of valid frames is MAX_NUM_FRAMES * MAX_NUM_PACKING.
MAX_NUM_PACKING=6  # indicates the maximum packing number of video frames. valid range: 1-6
TIME_SCALE = 0.1 
fps = 8 # fps for video
force_packing = None # You can set force_packing to ensure that 3D-Resampler packing is forcibly enabled; otherwise, encode_video will dynamically set the packing quantity based on the duration.
model_path = 'QwenModelPath' # You can replace it with your own model path.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
    device_map="auto",
    use_cache=True,
)
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
model = model.eval()
VIDEO_BASE_PATH = './eval_videos/{}.mp4'
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
def map_to_nearest_scale(values, scale):
        tree = cKDTree(np.asarray(scale)[:, None])
        _, indices = tree.query(np.asarray(values)[:, None])
        return np.asarray(scale)[indices]

def group_array(arr, size):
    return [arr[i:i+size] for i in range(0, len(arr), size)]

def encode_video(video_path, choose_fps=3, force_packing=None):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    video_duration = len(vr) / fps
        
    if choose_fps * int(video_duration) <= MAX_NUM_FRAMES:
        packing_nums = 1
        choose_frames = round(min(choose_fps, round(fps)) * min(MAX_NUM_FRAMES, video_duration))
        
    else:
        packing_nums = math.ceil(video_duration * choose_fps / MAX_NUM_FRAMES)
        if packing_nums <= MAX_NUM_PACKING:
            choose_frames = round(video_duration * choose_fps)
        else:
            choose_frames = round(MAX_NUM_FRAMES * MAX_NUM_PACKING)
            packing_nums = MAX_NUM_PACKING

    frame_idx = [i for i in range(0, len(vr))]      
    frame_idx =  np.array(uniform_sample(frame_idx, choose_frames))

    if force_packing:
        packing_nums = min(force_packing, MAX_NUM_PACKING)
    
    print(video_path, ' duration:', video_duration)
    print(f'get video frames={len(frame_idx)}, packing_nums={packing_nums}')
    
    frames = vr.get_batch(frame_idx).asnumpy()

    frame_idx_ts = frame_idx / fps
    scale = np.arange(0, video_duration, TIME_SCALE)

    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
    frame_ts_id = frame_ts_id.astype(np.int32)

    assert len(frames) == len(frame_ts_id)

    frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
    frame_ts_id_group = group_array(frame_ts_id, packing_nums)
    
    return frames, frame_ts_id_group

def single_request(prompt, video_path):
    # video_path = os.path.join(impossibe_video_root, video)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": fps,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        # fps=fps,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(output_text[0])
    return output_text[0]

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
    
    output_path = os.path.join(output_root, 'eval_res_Q7B_query_aligned.json')
    res_dict = {}
    if os.path.exists(output_path):
        res_dict = json.load(open(output_path, "r"))
    # res_dict = {}

    eval_keys = list(eval_content.keys())
    
    eval_number_mapping = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for eval_key in tqdm(eval_keys):
        if eval_key in res_dict:
            print('skip:', eval_key)
            continue
        try:
            video_path = VIDEO_BASE_PATH.format(eval_key)
            

            # Eval the overall
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
                output_response = single_request(
                    prompt=overall_prompt,
                    video_path=video_path
                )
                print('res:', output_response, 'res type:', type(output_response))
                if '<output>' not in output_response or '</output>' not in output_response:
                    continue
                match = re.search(r'<output>(.*?)</output>', output_response, re.DOTALL)

                geted_output = match.group(1)
                res = geted_output

                eval_content[eval_key]['t2v_eval_event_info']['overall_event_res'] = res
                print('filter res:', filter_and_join_letters(res))
                eval_content[eval_key]['t2v_eval_event_info']['overall_event_processed_res'] = filter_and_join_letters(res)
                wrong_output_flag = False


            # Eval the static
            verification_checks = eval_content[eval_key]['verification_checks']
            for v_check in verification_checks:
                if 'check' in v_check:
                    if v_check['check'] == False:
                        continue
                wrong_output_flag = True
                idx = 0
                while wrong_output_flag:
                    single_verification_check = EVAL_PROMPT_FOR_STATISTIC_QUESTION.format(v_check['question'])
                    res = single_request(
                        prompt=single_verification_check,
                        video_path=video_path
                    )
                    v_check['res'] = extract_first_yes_or_no(res)
                    idx += 1
                    if v_check['res'] != 'wrong' or idx==3:
                        wrong_output_flag=False
                

            res_dict[eval_key] = eval_content[eval_key]
            if len(list(res_dict.keys())) % 3 == 0:
                json.dump(res_dict, open(output_path, "w"), indent=4)
        except Exception as e:
            print('Error:', e)

    json.dump(res_dict, open(output_path, "w"), indent=4)
