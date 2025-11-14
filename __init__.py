import torch
import re
import comfy.model_management
import copy
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict

class CLIPTextEncodeAttentionBias(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s): -> InputTypeDict:
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    FUNCTION = "encode"
    CATEGORY = "conditioning"
    DESCRIPTION = "Prompt parser which applies attention bias instead of weights. Use '<' and '>' to enclose it and '=1.0' to specify attention strength. Example: 'This is <a huge dog=1.25>'."

    def _get_token_count(self, clip, text):
        """
        Robustly tokenizes a text segment and returns the number of its content tokens.
        """
        if not text:
            return 0
        
        tokens = clip.tokenize(text)
        
        max_content_len = 0
        for key in tokens:
            if len(tokens[key]) > 0 and len(tokens[key][0]) > 0:

                content_len = len(tokens[key][0]) - 2
                if content_len > max_content_len:
                    max_content_len = content_len
        
        return max(0, max_content_len)

    def encode(self, clip, text):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        if '<' not in text and '>' not in text and '=' not in text:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            return ([[cond, {"pooled_output": pooled}]], )

        bias_pattern = re.compile(r"<([^>]+)=([0-9.-]+)>")
        split_pattern = re.compile(r"(<[^>]+=[0-9.-]+>)")
        segments = split_pattern.split(text)

        clean_text = ""
        biases_to_apply = []

        current_token_index = 1 

        for segment in segments:
            if not segment:
                continue

            match = bias_pattern.fullmatch(segment)
            if match:
                bias_text, strength_str = match.groups()
                strength = float(strength_str)
                clean_text += bias_text
                num_tokens = self._get_token_count(clip, bias_text)
                
                if num_tokens > 0:
                    start_index = current_token_index
                    end_index = current_token_index + num_tokens
                    biases_to_apply.append({"start": start_index, "end": end_index, "strength": strength})
                
                current_token_index += num_tokens
            else:
                clean_text += segment
                num_tokens = self._get_token_count(clip, segment)
                current_token_index += num_tokens

        tokens = clip.tokenize(clean_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        
        if not biases_to_apply:
            return ([[cond, {"pooled_output": pooled}]], )

        cond_dict = {"pooled_output": pooled}
        n_text_tokens = cond.shape[1]
        device = cond.device
        dtype = torch.float16


        final_seq_len = n_text_tokens + 1
        attn_mask = torch.zeros((1, final_seq_len, final_seq_len), dtype=dtype, device=device)

        pooled_offset = 1

        for bias in biases_to_apply:
            strength = bias["strength"]
            attn_bias_value = torch.log(torch.tensor(strength, dtype=dtype, device=device))

            start = min(bias["start"] + pooled_offset, final_seq_len)
            end = min(bias["end"] + pooled_offset, final_seq_len)

            if start >= end:
                continue
            
            attn_mask[:, :, start:end] += attn_bias_value
            attn_mask[:, start:end, :] += attn_bias_value
        
        cond_dict["attention_mask"] = attn_mask
        cond_dict["attention_mask_img_shape"] = (1, 1)
            
        return ([[cond, cond_dict]],)

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeAttentionBias": CLIPTextEncodeAttentionBias,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeAttentionBias": "CLIP Text Encode (w Attention Bias)",

}

