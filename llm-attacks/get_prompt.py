import gc
import torch
import numpy as np

from llm_attacks import get_nonascii_toks
from llm_attacks.minimal_gcg.string_utils import PromptManager, load_conversation_template
from llm_attacks.minimal_gcg.opt_utils import load_model_and_tokenizer, get_filtered_cands
from llm_attacks.minimal_gcg.opt_utils import token_gradients, sample_control, get_logits, target_loss

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)


def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 32

    gen_config.do_sample = False    
    if gen_config.max_new_tokens > 50:
        print('WARNING: max_new_tokens > 32 may cause testing to slow down.')

    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids,
                                attention_mask=attn_masks, 
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id)[0]

    return output_ids

def get_completion(model, tokenizer, prompt_manager, instruction, device):
    input_ids = prompt_manager.get_input_ids(instruction=instruction).to(device)

    gen_config = model.generation_config
    gen_config.max_new_tokens = 256

    completion = tokenizer.decode((generate(model, tokenizer, input_ids, prompt_manager._assistant_role_slice, gen_config=gen_config))).strip()

    print(f"\nCompletion: {completion}")

def main():
    updated_instruction = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    target = "  The Tour de France is an annual bicycle race that's considered the most prestigious cycling event in the world"

    model_path = "meta-llama/Llama-2-7b-chat-hf"
    template_name = 'llama-2'
    device = 'cuda:1'

    allow_non_ascii = True
    batch_size = 128
    num_steps = 500
    topk = 256

    model, tokenizer = load_model_and_tokenizer(
        model_path, low_cpu_mem_usage=True,
        use_cache=False, device=device
    )
    conv_template = load_conversation_template(template_name)

    prompt_manager = PromptManager(
        tokenizer=tokenizer,
        conv_template=conv_template,
        instruction=updated_instruction,
        target=target
    )

    # print each slice of the prompt
    initial_input_ids = prompt_manager.get_input_ids()
    print(f"User Role Slice: {tokenizer.decode(initial_input_ids[prompt_manager._user_role_slice])}")
    print(f"Control Slice: {tokenizer.decode(initial_input_ids[prompt_manager._control_slice])}")
    print(f"Assistant Role Slice: {tokenizer.decode(initial_input_ids[prompt_manager._assistant_role_slice])}")
    print(f"Target Slice: {tokenizer.decode(initial_input_ids[prompt_manager._target_slice])}")
    print(f"Loss Slice: {tokenizer.decode(initial_input_ids[prompt_manager._loss_slice])}")

    not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)

    for _ in range(num_steps):

        input_ids = prompt_manager.get_input_ids(instruction=updated_instruction).to(device)

        coordinate_grad = token_gradients(
            model, input_ids,
            prompt_manager._control_slice,
            prompt_manager._target_slice,
            prompt_manager._loss_slice
        )

        with torch.no_grad():

            control_slice = prompt_manager._control_slice
            target_slice = prompt_manager._target_slice

            new_instruction_toks = sample_control(
                control_toks=input_ids[control_slice].to(device),
                grad=coordinate_grad, batch_size=batch_size, topk=topk,
                temp=1, not_allowed_tokens=not_allowed_tokens
            )

            del coordinate_grad

            new_instruction_toks = get_filtered_cands(
                tokenizer, new_instruction_toks,
                filter_cand=True, curr_control=updated_instruction
            )

            logits, ids = get_logits(
                model=model, tokenizer=tokenizer,
                input_ids=input_ids, control_slice=control_slice,
                test_controls=new_instruction_toks, return_ids=True,
                batch_size=128
            )

            losses = target_loss(logits, ids, target_slice)

            del ids, input_ids

            best_new_instruction_id = losses.argmin()
            updated_instruction = new_instruction_toks[best_new_instruction_id]

            current_loss = losses[best_new_instruction_id]
            print(f"\nCurrent Loss: {current_loss:.4f}")

            target_logits = logits[best_new_instruction_id, target_slice, :]
            for position in range(len(target_logits)):
                logits_at_position = target_logits[position, :]
                top5 = torch.topk(logits_at_position, 5).indices
                print(f"Top 5 tokens at position {position}: {tokenizer.decode(top5)}")
                print(f"Top 5 logits at position {position}: {logits_at_position[top5]}")

            print(f"\nCurrent Suffix:{updated_instruction}", end='\r')

        del logits, current_loss, losses, best_new_instruction_id, new_instruction_toks
        gc.collect()
        torch.cuda.empty_cache()

        get_completion(model, tokenizer, prompt_manager, updated_instruction, device)

if __name__ == '__main__':
    main()
