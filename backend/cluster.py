from data import combine_time_frames
from symbolic_seq.description_length import calculate_total_dl, calculate_data_dl

def generate_random_seeds():
    return [0]

def fold_time_frames(tfs, random_seeds, weight):
    folding_list = []
    folded_list = []

    for idx, tf in enumerate(tfs):
        if idx in random_seeds:
            if len(folding_list) > 0:
                new_tfs = combine_time_frames(folding_list)
                folded_list = folded_list + new_tfs
            folding_list = [tf]
        else:
            temp_folding_list = folding_list + [tf]
            temp_time_frame = combine_time_frames(temp_folding_list)
            if calculate_total_dl(temp_time_frame, weight) < sum(calculate_data_dl([tf], weight) for tf in temp_folding_list):
                folding_list = temp_folding_list
            else:
                new_tfs = combine_time_frames(folding_list)
                folded_list = folded_list + new_tfs
                folding_list = [tf]
        if idx == len(tfs) - 1:
            new_tfs = combine_time_frames(folding_list)
            folded_list = folded_list + new_tfs
    
    return folded_list

def fold_timeline(tfs, random_seeds, weight):
    folded_tfs = fold_time_frames(tfs, random_seeds, weight)
    if len(folded_tfs) == 1:
        return folded_tfs
    if len(folded_tfs) < len(tfs):
        random_seeds = generate_random_seeds()
        return fold_timeline(folded_tfs, random_seeds, weight)
    return folded_tfs

def find_mdl(tfs, weight, MAX_ITER=1):
    best_tfs = tfs
    min_dl = sum(calculate_data_dl([tf], weight) for tf in tfs)
    for i in range(MAX_ITER):
        random_seeds = generate_random_seeds()
        folded_tfs = fold_time_frames(tfs, random_seeds, weight)
        folded_dl = sum(calculate_data_dl([tf], weight) for tf in folded_tfs)
        if folded_dl < min_dl:
            min_dl = folded_dl
            best_tfs = folded_tfs
    return best_tfs, min_dl