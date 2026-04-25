import os
import sys
current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(os.path.join(current_path, "utils"))
import math
import pandas as pd
import argparse
from utils.output_utils import Tee


#####################################################################
# DiTPA hardware simulator
# (1) Simulate the DiTPA's hardware behavior to get the compute cycles and external memory access (ema) bytes
# (2) Consider the action, denoising and multimodality redundancy exploitation
# (3) Support baseline and DiTPA (action skip, denoising skip, multimodality skip) for comparison
# (4) Support pipeline or not for intra-block layers
#####################################################################

def reconfigurable_pe_array_sim():
    # PE configuration
    pe_size = args.pe_size # each PE supports 64 MACs
    pe_num_intra_line = args.pe_num_intra_line # each PE line contains 6 PEs
    pe_line_num = args.pe_line_num # each PE array has 12 PE lines
    pe_array_size = pe_size * pe_num_intra_line * pe_line_num # each PE array slice supports 4608 MACs
    pe_array_slice = args.pe_array_slice # 2 PE array slices in total

    # bandwidth configuration
    bw_off_chip = args.bw_off_chip # off-chip bandwidth (bytes/cycle) from dram to sram, weight and activation data share bandwidth
    bw_on_chip_act, bw_on_chip_wgt = args.bw_on_chip_act, args.bw_on_chip_wgt # onchip bandwidth for activation and weight from sram to pe array
    parallel_wgt, parallel_act = 12, 1 # weight reused 12 times, activation reused 1 times

    # layer configuration parameters
    act_h, act_w = args.act_h, args.act_w # layer activation height and width
    wgt_h, wgt_w = args.wgt_h, args.wgt_w # layer weight height and width
    mode = args.mode  # compute mode: mode 1: 'A-W-Col', 'A-A-Col'; mdoe 2: 'A-W-Row', 'A-A-Row'

    # Example:
    # read stage [from off-chip/on-chip mem], compute stage, write back stage [to on-chip/off-chip mem], pipeline stage, update act and wgt stage (take layer Q compute as example)
    # 1. read stage: (1) DRAM to SRAM: read act 12 lines with half hid. dim. (one tile), needs 4608/384=12 cycles; read wgt 1 col with half dim., needs 384/384=1 cycle; 
    #                (2) SRAM to PE array: load act to core, copy wgt 11 times and load to core; needs 64 cycles for act load, and wgt cycles are hidden by act load latency;
    #                (3) Total time: (12+1-12)+64=65 cycles, pipeline update, i.e. 12 cycles, can be hidden,since act onchip bandwdith<offchip bandwidth; start compute;
    # 2. compute stage: compute and get 12 partial results from 12 lines; needs 1+6+3=10 cycles, 1 for mul, 6 for intra-pe add1, 3 for inter-pe add2;
    # 3. write back stage: write back 12 results to the output buffer, needs 1 cycle; 
    # 4. pipeline stage: overlap read wgt, compute and write back; each cycle output 12 partial results; needs (768-1)=767 cycles;
    # 5. update act and wgt stage: read right side 12 lines of act, needs 4608/384=12 cycles; read wgt, needs 1 cycles; and continue pipeline compute and write back;
    # 6. change to next 12 lines: since wgt can be stored onchip, only need to read act, needs 12 cycles; SRAM to PE array needs 64 cycles; Can be hidden due to PP buffer design, so the cycle is 0; and continue pipeline compute and write back;
    # => single tile with half dim. cycles: ((12+1-12)+64)+10+1+(768-1)=843 cycles;
    # => single tile with all dim. cycles: 843*2=1686 cycles;
    # => single layer cycles: 1686+768*2*(153/12-1)=20118 cycles;

    # Read stage and write back stage latency analysis:
    # 1. According to the analysis, for different input length compute, the latency for read stage and write back stage are same,
    # the main difference is in pipeline compute stage.
    # 2. Although the initial read stage latency can be hiddened in subsequent layers compute, 
    # no need for small latency reduction but increasing control complexity.

    stages = ['read', 'compute', 'write_back', 'pipeline', 'update_act']
    cycles = {stage: 0 for stage in stages}
   
    # single tile processing
    if mode == 'A-W-Col': # Q, K, V, O, G, U => mode 1 in the paper
        if args.op == 'q' or args.op == 'k' or args.op == 'v':  # Q,K,V read from DRAM, need ema
            cycles['read'] = (pe_array_size/bw_off_chip) + (pe_array_size/(bw_off_chip*parallel_wgt)) + (pe_array_size/bw_on_chip_act) - (pe_array_size/bw_off_chip) # read act and wgt, and remove overlap time
        else: # O,G,U read from SRAM
            cycles['read'] = (pe_array_size/bw_on_chip_act)
        cycles['compute'] = 1 + math.ceil(math.log2(pe_size)) + math.ceil(math.log2(pe_num_intra_line))  # mul, add1, add2
        cycles['write_back'] = 1
        cycles['pipeline'] = wgt_w
        cycles['update_act'] = 0 # can be hidden
    elif mode == 'A-A-Col': # QK => mode 1 in the paper
        cycles['read'] = pe_array_size/bw_on_chip_act  # remove ema
        cycles['compute'] = 1 + math.ceil(math.log2(pe_size)) + math.ceil(math.log2(pe_num_intra_line))
        cycles['write_back'] = 1
        cycles['pipeline'] = wgt_w
        cycles['update_act'] = 0
    elif mode == 'A-W-Row' or mode == 'A-A-Row': # SV,D => mode 2 in the paper
        cycles['read'] = 1  # first read act and wgt, needs 1 cycle
        cycles['compute'] = 1 + 1 # mul, element-wise add
        cycles['write_back'] = 1
        cycles['pipeline'] = wgt_h
        cycles['update_act'] = 0

    # single layer processing
    act_update_times = math.ceil(act_h / (pe_num_intra_line*2))-1  # *2 for left half dim. and right half dim.; round up to match hardware behavior
    for stage in stages:
        if stage in ['read', 'compute', 'write_back', 'update_act']:
            if args.op == 'q' or args.op == 'k' or args.op == 'v':
                cycles[stage] = cycles[stage]*2  # *2 for left half dim. and right half dim.
            else:
                cycles[stage] = cycles[stage] # store onchip without ema
        if stage in ['pipeline']:
            cycles[stage] = (cycles[stage]-1)*2 + cycles[stage]*2*act_update_times
    total_cycles = sum(cycles.values())

    num_act, num_wgt = 0, 0
    if args.op == 'q' or args.op == 'k' or args.op == 'v': # activation ema
        num_act = act_h * act_w
    else:
        num_act = 0
    
    if args.op == 'q' or args.op == 'k' or args.op == 'v' or args.op == 'o' or args.op == 'g' or args.op == 'u' or args.op == 'd': # weigt ema
        num_wgt = wgt_h * wgt_w
    else:
        num_wgt = 0
    total_bytes = (num_act + num_wgt)

    return total_cycles, total_bytes


def sfu_sim():
    # configurations
    bw = args.bw_on_chip_wgt # bandwidth (bytes/cycle)
    sfu_type = args.sfu_type # sfu operation type
    act_num, act_h, act_w = args.act_num, args.act_h, args.act_w # layer activation head number, height and width

    # sfu type: rms, rope, scale, mask, softmax, resadd, silu, elemul
    # stages: read stage [from on-chip mem], compute stage, write back stage [to on-chip mem]
    stages = ['read', 'compute', 'write_back']
    cycles = {stage: 0 for stage in stages}

    if sfu_type == 'rms':  # line-wise
        cycles['read'] = 1  # 1 for act with 12 data
        first_line_cycles = 15 # first line for act_w = 768, compute cycles for initial first line
        pipelined_line_cycles = (first_line_cycles-1) + 14 # every 14 lines, compute cycles for multiple pipelined lines
        cycles['compute'] = (act_h//14)*pipelined_line_cycles + ((first_line_cycles-1)+(act_h%14)) if act_h%14 !=0 else (act_h//14)*pipelined_line_cycles
        cycles['write_back'] = 1
    elif sfu_type == 'rope':  # line-wise
        cycles['read'] = 1 + math.ceil(2*act_h*(act_w/act_num)/bw) # 1 for act, other for paras
        first_line_cycles = 3 # first line for act_w = 768
        pipelined_line_cycles = (first_line_cycles-1) + 2 # every 2 lines
        cycles['compute'] = (act_h//2)*pipelined_line_cycles + ((first_line_cycles-1)+(act_h%2)) if act_h%2 !=0 else (act_h//2)*pipelined_line_cycles
        cycles['write_back'] = 1
    elif sfu_type == 'scale': # row-wise
        cycles['read'] = 1
        first_line_cycles = 1 # first line for 4*act_w =612
        pipelined_line_cycles = 1 # every line
        cycles['compute'] = (first_line_cycles-1) + act_h*(act_num//4)*pipelined_line_cycles
        cycles['write_back'] = 1
    elif sfu_type == 'mask':
        cycles['read'] = 1
        first_line_cycles = 1 # first line for 4*act_w =612
        pipelined_line_cycles = 1 # every line
        cycles['compute'] = (first_line_cycles-1) + act_h*(act_num//4)*pipelined_line_cycles
        cycles['write_back'] = 1
    elif sfu_type == 'softmax':  # row-wise
        cycles['read'] = 1
        first_line_cycles = 13 # first line for 4*act_w = 612
        pipelined_line_cycles = 1 # every line
        cycles['compute'] = (first_line_cycles-1) + act_h*(act_num//4)*pipelined_line_cycles
        cycles['write_back'] = 1
    elif sfu_type == 'resadd':  # row-wise
        cycles['read'] = 1
        first_line_cycles = 1 # first line for act_w = 768
        pipelined_line_cycles = 1 # every line
        cycles['compute'] = (first_line_cycles-1) + act_h*pipelined_line_cycles
        cycles['write_back'] = 1
    elif sfu_type == 'silu':  # row-wise
        cycles['read'] = 1
        first_line_cycles = 3 # first line for act_w = 768
        pipelined_line_cycles = (first_line_cycles-1) + 2 # every 2 lines
        cycles['compute'] = (act_h*3//2)*pipelined_line_cycles + ((first_line_cycles-1)+(act_h%2)) if act_h%2 !=0 else (act_h*3//2)*pipelined_line_cycles
        cycles['write_back'] = 1
    elif sfu_type == 'elemul':  # row-wise
        cycles['read'] = 1
        first_line_cycles = 1 # first line for act_w = 768
        pipelined_line_cycles = 1 # every line
        cycles['compute'] = (first_line_cycles-1) + act_h*3*pipelined_line_cycles
        cycles['write_back'] = 1
    total_cycles = sum(cycles.values())

    return total_cycles


def pipeline_sim(df=None, multimodality_skip=False):
    # consider layer-wise pipeline, while the intra-layer pipeline has beed processed in reconfigurable_pe_array_sim and sfu_sim
    # pipeline mm compute and sfu compute, thus to hidden sfu compute latency and memory access: 
    # => compare current mm compute latency with subsequent sfu compute latency, 
    # if mm > sfu, then hidden sfu latency, else need extra cycles to wait for sfu

    # pipeline flow:
    # 1. divide the layers into groups with mm and sfu
    # 2. for each group, process the continuous mm and sfu
    # 3. finally, calculate the total cycles

    if df is None:
        df = pd.read_excel(args.net_config_path)
    
    cycles_block = 0
    bytes_block = 0
    for g in range(df.iloc[df.shape[0]-1, 1]): # group-wise processing
        group_df = df[df['group'] == g+1]  # divide the layers into groups

        cycles_group, mm_cycles, sfu_cycles = 0, 0, 0
        mm_bytes = 0
        for i in range(group_df.shape[0]): # layer-wise processing
            args.op_type = group_df.iloc[i, 3] 
            args.op = group_df.iloc[i, 4]
            args.act_num = group_df.iloc[i, 5]
            args.act_h = group_df.iloc[i, 6]
            args.act_w = group_df.iloc[i, 7]
            args.wgt_h = group_df.iloc[i, 8]
            args.wgt_w = group_df.iloc[i, 9]
            args.mode = group_df.iloc[i, 10]
            args.sfu_type = group_df.iloc[i, 11]

            if args.op_type == 'mm':
                single_mm_cycles, single_mm_bytes = reconfigurable_pe_array_sim()
                # action col sparsity
                if multimodality_skip and args.op == 'sv': 
                    single_mm_cycles = single_mm_cycles * 0.52
                elif multimodality_skip and args.op == 'o':
                    single_mm_cycles = single_mm_cycles * 0.52
                    
                mm_cycles += single_mm_cycles
                mm_bytes += single_mm_bytes
            elif args.op_type == 'sfu':
                single_sfu_cycles = sfu_sim()
                sfu_cycles += single_sfu_cycles
        
        # pipeline mm and sfu, hidden sfu latency
        if mm_cycles > sfu_cycles:  
            cycles_group = mm_cycles
        else:
            cycles_group = sfu_cycles
        
        cycles_block += cycles_group
        bytes_block += mm_bytes
        
    return cycles_block, bytes_block


def block_sim(df=None, layer_pipeline=False, multimodality_skip=False):
    cycles_block = 0
    bytes_block = 0
    if layer_pipeline:
        # single block processing with layer-wise pipeline
        cycles_block, bytes_block = pipeline_sim(df=df, multimodality_skip=multimodality_skip)
    else:
        # single block processing without layer-wise pipeline
        for i in range(df.shape[0]): # layer-wise processing
            args.op_type = df.iloc[i, 3]
            args.op = df.iloc[i, 4]
            args.act_num = df.iloc[i, 5]
            args.act_h = df.iloc[i, 6]
            args.act_w = df.iloc[i, 7]
            args.wgt_h = df.iloc[i, 8]
            args.wgt_w = df.iloc[i, 9]
            args.mode = df.iloc[i, 10]
            args.sfu_type = df.iloc[i, 11]

            cycles_layer = 0
            bytes_layer = 0
            if args.op_type == 'mm':
                cycles_layer, bytes_layer = reconfigurable_pe_array_sim()
                # action col sparsity
                if multimodality_skip and args.op == 'sv':  
                    cycles_layer = cycles_layer * 0.52
                elif multimodality_skip and args.op == 'o':  
                    cycles_layer = cycles_layer * 0.52
            elif args.op_type == 'sfu':
                cycles_layer = sfu_sim()

            cycles_block += cycles_layer
            bytes_block += bytes_layer

    return cycles_block, bytes_block


def scheduler(df=None, layer_pipeline=False, action_skip=False, denoising_skip=False, multimodality_skip=False, efficiency=False):
    # load block configurations
    if df is None:
        df = pd.read_excel(args.net_config_path) 

    df_origin = df.iloc[0:20]
    df_l_skip = df.iloc[20:40]
    df_lv1_skip = df.iloc[40:60]
    df_lv1v2_skip = df.iloc[60:80]

    # compute block cycles and ema bytes
    cycles_block_origin, bytes_block_origin = block_sim(df=df_origin, layer_pipeline=layer_pipeline, multimodality_skip=multimodality_skip)
    cycles_block_l_skip, bytes_block_l_skip = block_sim(df=df_l_skip, layer_pipeline=layer_pipeline, multimodality_skip=multimodality_skip)
    cycles_block_lv1_skip, bytes_block_lv1_skip = block_sim(df=df_lv1_skip, layer_pipeline=layer_pipeline, multimodality_skip=multimodality_skip)
    cycles_block_lv1v2_skip, bytes_block_lv1v2_skip = block_sim(df=df_lv1v2_skip, layer_pipeline=layer_pipeline, multimodality_skip=multimodality_skip)
    
    # compute total cycles and ema bytes according to the redundancy exploitation strategy
    block_num = args.block_num
    iter_num = args.iter_num
    iter_num_after_skip = iter_num - args.iter_skip_num
    action_num = args.ditpa_action_num
    action_num_after_skip = action_num - args.ditpa_action_skip_num

    if action_skip and denoising_skip and multimodality_skip:
        # total action skip cycles
        action_skip_judge_cycles = 4 # from RTL simulation
        cycles_action_skip = (action_num - action_num_after_skip) * action_skip_judge_cycles

        # total denoising skip cycles
        iter_df = df_lv1v2_skip[df_lv1v2_skip['operator'] == 'resadd']  # get the residual layer
        cycles_block_iter_skip = 0
        bytes_block_iter_skip = 0
        for i in range(iter_df.shape[0]): # layer-wise processing
            args.op_type = iter_df.iloc[i, 3]
            args.op = iter_df.iloc[i, 4]
            args.act_num = iter_df.iloc[i, 5]
            args.act_h = iter_df.iloc[i, 6]
            args.act_w = iter_df.iloc[i, 7]
            args.wgt_h = iter_df.iloc[i, 8]
            args.wgt_w = iter_df.iloc[i, 9]
            args.mode = iter_df.iloc[i, 10]
            args.sfu_type = iter_df.iloc[i, 11]
            cycles_block_iter_skip += sfu_sim() * 2
            bytes_block_iter_skip += args.act_h * args.act_w
        cycles_iter_skip = cycles_block_iter_skip * block_num * (iter_num - iter_num_after_skip) * action_num_after_skip
        bytes_iter_skip = bytes_block_iter_skip * block_num * (iter_num - iter_num_after_skip) * action_num_after_skip

        # total multimodality skip cycles and ema bytes
        # feature map buffering considered in the configuration table, directly loading buffered K and V features
        # action column sparsity considerd in the block_sim/pipeline_sim
        iter_num_origin = 1
        iter_num_l_skip = action_num - action_num_after_skip
        iter_num_lv1_skip = action_num_after_skip - iter_num_l_skip - 1
        iter_num_lv1v2_skip = action_num_after_skip * (iter_num_after_skip - 1)
       
        cycles_origin = cycles_block_origin * block_num * iter_num_origin
        cycles_l_skip = cycles_block_l_skip * block_num * iter_num_l_skip
        cycles_lv1_skip = cycles_block_lv1_skip * block_num * iter_num_lv1_skip
        cycles_lv1v2_skip = cycles_block_lv1v2_skip * block_num * iter_num_lv1v2_skip
        cycles_modality_skip = cycles_origin + cycles_l_skip + cycles_lv1_skip + cycles_lv1v2_skip

        bytes_origin = bytes_block_origin * block_num * iter_num_origin
        bytes_l_skip = bytes_block_l_skip * block_num * iter_num_l_skip
        bytes_lv1_skip = bytes_block_lv1_skip * block_num * iter_num_lv1_skip
        bytes_lv1v2_skip = bytes_block_lv1v2_skip * block_num * iter_num_lv1v2_skip
        bytes_modality_skip = bytes_origin + bytes_l_skip + bytes_lv1_skip + bytes_lv1v2_skip

        # total cycles and ema bytes
        cycles_all_action = cycles_action_skip + cycles_iter_skip + cycles_modality_skip
        cycles_single_action = cycles_all_action / action_num
        bytes_all_action = bytes_iter_skip + bytes_modality_skip

        # power consumption
        power = args.total_power * 1e-3
    elif action_skip and denoising_skip:
        # total action skip cycles
        action_skip_judge_cycles = 4
        cycles_action_skip = (action_num - action_num_after_skip) * action_skip_judge_cycles

        # total denoising skip cycles
        iter_df = df_origin[df_origin['operator'] == 'resadd'] 
        cycles_block_iter_skip = 0
        bytes_block_iter_skip = 0
        for i in range(iter_df.shape[0]):
            args.op_type = iter_df.iloc[i, 3] 
            args.op = iter_df.iloc[i, 4]
            args.act_num = iter_df.iloc[i, 5]
            args.act_h = iter_df.iloc[i, 6]
            args.act_w = iter_df.iloc[i, 7]
            args.wgt_h = iter_df.iloc[i, 8]
            args.wgt_w = iter_df.iloc[i, 9]
            args.mode = iter_df.iloc[i, 10]
            args.sfu_type = iter_df.iloc[i, 11]
            cycles_block_iter_skip += sfu_sim() * 2
            bytes_block_iter_skip += args.act_h * args.act_w
        cycles_iter_skip = cycles_block_iter_skip * block_num * (iter_num - iter_num_after_skip) * action_num_after_skip
        bytes_iter_skip = bytes_block_iter_skip * block_num * (iter_num - iter_num_after_skip) * action_num_after_skip

        # total cycles and ema bytes
        cycles_common_compute = cycles_block_origin * block_num * iter_num_after_skip * action_num_after_skip
        cycles_all_action = cycles_action_skip + cycles_iter_skip + cycles_common_compute
        cycles_single_action = cycles_all_action / action_num

        bytes_common_compute = bytes_block_origin * block_num * iter_num_after_skip * action_num_after_skip
        bytes_all_action = bytes_iter_skip + bytes_common_compute
        
        # power consumption
        power = args.total_power * 1e-3
    elif action_skip:
        # total action skip cycles
        action_skip_judge_cycles = 4
        cycles_action_skip = (action_num - action_num_after_skip) * action_skip_judge_cycles
        
        # total cycles and ema bytes
        cycles_all_action = (cycles_block_origin * block_num * iter_num * action_num_after_skip) + cycles_action_skip
        cycles_single_action = cycles_all_action / action_num

        bytes_all_action = bytes_block_origin * block_num * iter_num * action_num_after_skip
        
        # power consumption
        power = (args.total_power - args.data_manager_power) * 1e-3 
    elif denoising_skip:
        # single block denoising skip cycles and ema bytes
        iter_df = df_origin[df_origin['operator'] == 'resadd'] 
        cycles_block_iter_skip = 0
        bytes_block_iter_skip = 0
        for i in range(iter_df.shape[0]): 
            args.op_type = iter_df.iloc[i, 3] 
            args.op = iter_df.iloc[i, 4]
            args.act_num = iter_df.iloc[i, 5]
            args.act_h = iter_df.iloc[i, 6]
            args.act_w = iter_df.iloc[i, 7]
            args.wgt_h = iter_df.iloc[i, 8]
            args.wgt_w = iter_df.iloc[i, 9]
            args.mode = iter_df.iloc[i, 10]
            args.sfu_type = iter_df.iloc[i, 11]
            cycles_block_iter_skip += sfu_sim() * 2
            bytes_block_iter_skip += args.act_h * args.act_w
        
        # total cycles and ema bytes
        cycles_all_iter_skip = cycles_block_iter_skip * block_num * (iter_num - iter_num_after_skip)
        cycles_all_iter_origin = cycles_block_origin * block_num * iter_num_after_skip
        cycles_all_action = (cycles_all_iter_skip + cycles_all_iter_origin) * action_num
        cycles_single_action = cycles_all_action / action_num 

        bytes_all_iter_skip = bytes_block_iter_skip * block_num * (iter_num - iter_num_after_skip)
        bytes_all_iter_origin = bytes_block_origin * block_num * iter_num_after_skip
        bytes_all_action = (bytes_all_iter_skip + bytes_all_iter_origin) * action_num

        # power consumption
        power = (args.total_power - args.action_predictor_power) * 1e-3
    elif multimodality_skip:
        # total cycles and ema bytes of different modality 
        iter_num_origin = 1
        iter_num_lv1_skip = action_num - 1
        iter_num_lv1v2_skip = action_num*(iter_num-1)

        cycles_origin = cycles_block_origin * block_num * iter_num_origin
        cycles_lv1_skip = cycles_block_lv1_skip * block_num * iter_num_lv1_skip
        cycles_lv1v2_skip = cycles_block_lv1v2_skip * block_num * iter_num_lv1v2_skip

        bytes_origin = bytes_block_origin * block_num * iter_num_origin
        bytes_lv1_skip = bytes_block_lv1_skip * block_num * iter_num_lv1_skip
        bytes_lv1v2_skip = bytes_block_lv1v2_skip * block_num * iter_num_lv1v2_skip

        # total cycles and ema bytes
        cycles_all_action = cycles_origin + cycles_lv1_skip + cycles_lv1v2_skip
        cycles_single_action = cycles_all_action / action_num
        bytes_all_action = bytes_origin + bytes_lv1_skip + bytes_lv1v2_skip

        # power consumption
        power = (args.total_power - args.action_predictor_power) * 1e-3
    else:
        # total cycles and ema bytes
        cycles_all_action = cycles_block_origin * block_num * iter_num * action_num
        cycles_single_action = cycles_all_action / action_num

        bytes_all_action = bytes_block_origin * block_num * iter_num * action_num

        # power consumption
        power = (args.total_power - args.data_manager_power - args.action_predictor_power) * 1e-3

    # performance evaluation
    action_freq, task_time, action_freq_speedup, task_time_speedup = performance_speedup_sim(cycles_single_action=cycles_single_action, cycles_all_action=cycles_all_action)

    if efficiency:
        efficiency_speedup = energy_efficiency_sim(total_time=task_time, total_bytes=bytes_all_action, power=power)
    
    return [action_freq, task_time, action_freq_speedup, task_time_speedup, efficiency_speedup if efficiency else None]


def performance_speedup_sim(cycles_single_action=None, cycles_all_action=None):
    target_freq = args.target_freq
    pe_array_slice = args.pe_array_slice # 2 PE array slices

    # baseline action frequency and task time
    baseline_action_freq = args.baseline_action_num / args.baseline_task_time
    baseline_task_time = args.baseline_task_time

    # DiTPA action frequency and task time
    ditpa_action_freq = (target_freq / cycles_single_action) * pe_array_slice
    ditpa_task_time = (cycles_all_action / target_freq) / pe_array_slice

    # speedup 4.05815972 
    scale = 4.05815972
    scale2gpu_freq = ditpa_action_freq * scale
    action_freq_speedup = scale2gpu_freq / baseline_action_freq   

    scale2gpu_task_time = ditpa_task_time / scale
    task_time_speedup = baseline_task_time / scale2gpu_task_time

    return ditpa_action_freq, ditpa_task_time, action_freq_speedup, task_time_speedup


def energy_efficiency_sim(total_time=None, total_bytes=None, power=None):
    # ditpa logic energy
    ditpa_logic_energy = power*total_time

    # dram energy
    dram_energy_per_bit = 0.85*1e-12  # 0.85 pJ
    dram_access_bits = total_bytes * 8
    dram_energy = dram_access_bits * dram_energy_per_bit  # in pJ

    # total operations
    ops_single_action = 0.6715 # in TOPs
    total_ops = ops_single_action * args.ditpa_action_num

    # energy efficiency gains
    gpu_avg_power = 80 # in W
    gpu_efficiency = (ops_single_action*args.baseline_action_num) / (gpu_avg_power * args.baseline_task_time)
    efficiency_speedup = (total_ops) / (ditpa_logic_energy + dram_energy) / gpu_efficiency
    
    return efficiency_speedup


def ditpa_evaluation():    
    # read configs
    net_structure = pd.read_excel(args.net_config_path)
    baseline_sw_res = pd.read_excel(args.baseline_sw_res_path)
    ditpa_sw_res = pd.read_excel(args.ditpa_sw_res_path)

    baseline_success_rate_list = baseline_sw_res['success_rate'].tolist()
    baseline_actions_list = baseline_sw_res['actions'].tolist()
    baseline_task_time_list = baseline_sw_res['task_time'].tolist()
    baseline_pi_list = baseline_sw_res['position_instability'].tolist()
    baseline_vi_list = baseline_sw_res['velocity_instability'].tolist()
    
    ditpa_sucess_rate_list = ditpa_sw_res['success_rate'].tolist()
    ditpa_total_actions_list = ditpa_sw_res['total_actions'].tolist()
    ditpa_skip_actions_list = ditpa_sw_res['skip_actions'].tolist()
    ditpa_pi_list = ditpa_sw_res['position_instability'].tolist()
    ditpa_vi_list = ditpa_sw_res['velocity_instability'].tolist()

    # task-wise evaluation
    for i, (ditpa_action_num, ditpa_action_skip_num) in enumerate(zip(ditpa_total_actions_list, ditpa_skip_actions_list)):
        args.baseline_action_num = baseline_actions_list[i]
        args.baseline_task_time = baseline_task_time_list[i]
        args.ditpa_action_num = ditpa_action_num
        args.ditpa_action_skip_num = ditpa_action_skip_num

        if i == 10: # average
            print(f"\n================================================ Average Evaluation ================================================")
        else:
            print(f"\n================================================ Task {i+1} Evaluation ================================================")
        print(f"----------------------------------------- (1) Accuracy ---------------------------------------")
        print(f"Baseline Success Rate: {baseline_success_rate_list[i]*100:.0f}%; DiTPA Success Rate: {ditpa_sucess_rate_list[i]*100:.0f}%")
        print(f"Baseline Position Instability: {baseline_pi_list[i]:.3f}; DiTPA Position Instability: {ditpa_pi_list[i]:.3f}")
        print(f"Baseline Velocity Instability: {baseline_vi_list[i]:.3f}; DiTPA Velocity Instability: {ditpa_vi_list[i]:.3f}")

        print(f"----------------------------------------- (2) Performance -----------------------------------")
        res_performance = scheduler(df=net_structure, layer_pipeline=True, action_skip=True, denoising_skip=True, multimodality_skip=True, efficiency=True)
        print(f"Action Frequency: {res_performance[0]:.2f}Hz; Normalized Speedup: {res_performance[2]:.2f}x")
        print(f"Task Time: {res_performance[1]:.2f}s; Normalized Speedup: {res_performance[3]:.2f}x")
        print(f"Normalized energy efficiency speedup: {res_performance[4]:.2f}x")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # config path
    parser.add_argument('--net_config_path', type=str, default='./config/action_planner_structure.xlsx', help='Path to the net configuration file')
    parser.add_argument('--baseline_sw_res_path', type=str, default='./outputs/example_output/baseline_software_results.xlsx', help='Path to the baseline software results file')
    parser.add_argument('--ditpa_sw_res_path', type=str, default='./outputs/example_output/ditpa_software_results.xlsx', help='Path to the DiTPA software results file')
    parser.add_argument('--eval_res_out_path', type=str, default='./results/ditpa_evaluation_results.txt', help='Path to the DiTPA evaluation results file')
    # action planner configurations
    parser.add_argument('--op_type', type=str, default='mm', help='Operator type (mm/sfu)')
    parser.add_argument('--op', type=str, default='mm', help='Operator')
    parser.add_argument('--act_num', type=float, default=1, help='Activation head number')
    parser.add_argument('--act_h', type=float, default=153, help='Activation height')
    parser.add_argument('--act_w', type=float, default=768, help='Activation width')
    parser.add_argument('--wgt_h', type=float, default=768, help='Weight height')
    parser.add_argument('--wgt_w', type=float, default=768, help='Weight width')
    parser.add_argument('--sfu_type', type=str, default='0', help='rms, rope, scale, mask, softmax, resadd, silu, elemul')
    parser.add_argument('--block_num', type=float, default=12, help='Block number')
    parser.add_argument('--iter_num', type=float, default=50, help='Iteration number')
    # software intermediate results
    parser.add_argument('--baseline_action_num', type=float, default=380.05, help='Baseline action number')
    parser.add_argument('--baseline_task_time', type=float, default=166.64, help='Baseline task time')
    parser.add_argument('--ditpa_action_num', type=float, default=377.03, help='Action number')
    parser.add_argument('--ditpa_action_skip_num', type=float, default=159.4, help='Action skip number')
    # hardware configurations
    parser.add_argument('--target_freq', type=float, default=500e6, help='Target frequency')
    parser.add_argument('--total_power', type=float, default=1046.1242, help='Total power consumption')
    parser.add_argument('--action_predictor_power', type=float, default=0.27, help='Action predictor power consumption')
    parser.add_argument('--data_manager_power', type=float, default=14.77, help='Data manager power consumption')
    parser.add_argument('--iter_skip_num', type=float, default=20, help='Iteration skip number')
    parser.add_argument('--mode', type=str, default='A-W-Col', help='mode1 for A-W-Col, A-A-Col; mdoe 2 for A-W-Row, A-A-Row')
    parser.add_argument('--pe_size', type=float, default=64, help='PE size')
    parser.add_argument('--pe_num_intra_line', type=float, default=6, help='PE number in each line')
    parser.add_argument('--pe_line_num', type=float, default=12, help='PE line number')
    parser.add_argument('--pe_array_slice', type=float, default=2, help='PE array slice')
    parser.add_argument('--bw_off_chip', type=float, default=384, help='Off-chip bandwidth')
    parser.add_argument('--bw_on_chip_act', type=float, default=72, help='On-chip activation bandwidth')
    parser.add_argument('--bw_on_chip_wgt', type=float, default=384, help='On-chip weight bandwidth')
    args = parser.parse_args() 
    
    with open(args.eval_res_out_path, "w", encoding="utf-8") as f:
        tee_out = Tee(sys.__stdout__, f)
        old_out = sys.stdout
        try:
            sys.stdout = tee_out
            ditpa_evaluation()
        finally:
            sys.stdout = old_out
