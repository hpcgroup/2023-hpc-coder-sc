from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorForSeq2Seq
from datasets import load_dataset
from axonn.models.transformers import parallelize 
from axonn import axonn as ax
import torch
import random
import numpy as np
from arguments import create_parser
from contextlib import nullcontext
from data_utils import get_tokenizer_mapping_fn
from torch.utils.data import DataLoader
from axonn.checkpoint import save
from axonn.intra_layer import sync_gradients, optimize_communication, clear_weights_cache, clip_grad_norm_
from axonn import axonn as ax

def init_everything():
    torch.distributed.init_process_group(backend='nccl')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if rank == 0:
        print(f"Going to distribute the model over {world_size} GPUs")
    ax.init(G_data=1, G_inter=1, G_intra_r=1, G_intra_c=1, G_intra_d=world_size)

def set_seed(seed=123456):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


dtype_map = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32
}


def get_tokenized_dataset(tokenizer, sequence_length=256):
    data = load_dataset("hpcgroup/hpc-data")
    mapping_fn = get_tokenizer_mapping_fn(tokenizer, cutoff_len=sequence_length, train_on_inputs=False)
    train_data = data["train"].shuffle().map(mapping_fn, remove_columns=data["train"].column_names)
    return train_data

def pretty_log(iteration,
               total_train_iters,
               train_loss,
               elapsed_time_per_iteration,
               learning_rate,
               grad_norm,
):

    log_string = '> global batch {:8d}/{:8d} |'.format(
        iteration, total_train_iters)
    log_string += ' elapsed time per global batch (ms): {:.1f} |'.format(
        elapsed_time_per_iteration)
    log_string += ' learning rate: {:.3E} |'.format(learning_rate)
    log_string += ' loss: {:.5f} |'.format(train_loss)
    curr_mem =  torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    peak_mem =  torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    log_string += ' memory used by tensors {:.3f} GB (peak {:.3f} GB) |'.format(curr_mem, peak_mem)
    log_string += f' grad norm: {grad_norm:.5f}'
    return log_string


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    init_everything()
    set_seed(args.seed)
    dtype = dtype_map[args.dtype]

    if args.use_axonn:
        with parallelize(args.model_id):
            model = AutoModelForCausalLM.from_pretrained(args.model_id, 
                                                             torch_dtype=dtype, 
                                                         attn_implementation='eager' if not args.use_flash_attention else "flash_attention_2").to('cuda').float()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id,
                                                             torch_dtype=dtype,
                                                         attn_implementation='eager' if not args.use_flash_attention else "flash_attention_2").to('cuda').float()

    model.train()
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token_id = (
        0  
    )
    tokenizer.padding_side = "left"  
    tokenized_dataset = get_tokenized_dataset(tokenizer, args.sequence_length)
    sampler = torch.utils.data.distributed.DistributedSampler(
        tokenized_dataset
    )
    assert args.global_batch_size % (args.gradient_acc_steps * torch.distributed.get_world_size()) == 0
    dataloader = DataLoader(
                                tokenized_dataset, 
                                batch_size=args.global_batch_size // args.gradient_acc_steps // torch.distributed.get_world_size(), 
                                collate_fn=DataCollatorForSeq2Seq(
                                        tokenizer, 
                                        max_length=args.sequence_length if args.check_max_mem_usage else None,
                                        pad_to_multiple_of=8 if not args.check_max_mem_usage else None, 
                                        return_tensors="pt", 
                                        padding='max_length' if args.check_max_mem_usage else True
                                        ),
                                sampler=sampler
                            ) 
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=1e-5, 
                                  betas=(0.9, 0.95), 
                                  eps=1e-5,
                                  weight_decay=0.0)

    iters_per_epoch = len(dataloader) // args.gradient_acc_steps
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters_per_epoch * args.num_epochs)
    warmup_iters = 100
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_iters
            )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    loss_fn = torch.nn.CrossEntropyLoss()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    iter_no = 0
    for epoch_no in range(args.num_epochs):
        microbatch_no = 0
        start_event.record()
        batch_loss = 0
        for batch in dataloader:
            input_ids, labels, attention_mask = batch["input_ids"], batch["labels"], batch["attention_mask"]
            input_ids, labels, attention_mask = input_ids.cuda(), labels.cuda(), attention_mask.cuda()
            with optimize_communication(True, True, True, model):
                with torch.amp.autocast(device_type='cuda', dtype=dtype):
                    input_ids = input_ids[:, :-1]
                    attention_mask = attention_mask[:, :-1]
                    labels = labels[:, 1:]
                    output = model(input_ids = input_ids, attention_mask=attention_mask)
                    logits = output["logits"]
                    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), 
                                   labels.reshape(-1))
                scaler.scale(loss / args.gradient_acc_steps / torch.distributed.get_world_size()).backward()
            clear_weights_cache()
            global_loss = loss / args.gradient_acc_steps / torch.distributed.get_world_size()
            torch.distributed.all_reduce(global_loss)
            batch_loss += global_loss.item()
            microbatch_no += 1

            if microbatch_no == args.gradient_acc_steps:
                scaler.unscale_(optimizer)
                sync_gradients(model)
                grad_norm = clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                iter_no += 1
                end_event.record()
                if torch.distributed.get_rank() == 0 and (iter_no % args.log_interval==0):
                    torch.cuda.synchronize()
                    elapsed_time = start_event.elapsed_time(end_event) 
                    log_string = pretty_log(iter_no, len(dataloader)*args.num_epochs // args.gradient_acc_steps, 
                                            batch_loss, elapsed_time, learning_rate=optimizer.param_groups[0]['lr'], 
                                            grad_norm=grad_norm)
                    print(log_string)

                microbatch_no = 0
                batch_loss = 0
                start_event.record()
                state = {
                        "iter_no": iter_no,
                        "optimizer": optimizer.state_dict(),
                        "model": model.state_dict()
                        }
        save(state, checkpoint_folder="/pscratch/sd/a/amanc/ckpt", checkpoint_name=f"epoch_{epoch_no}")
