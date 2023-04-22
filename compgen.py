import re

def find_partition_name(name, lf):
    if lf == "cogs":
        return name
    else:
        return name+f"_{lf}"
    
def check_equal(left_lf, right_lf):
    index_mapping = {}
    current_idx = 0
    for t in left_lf.split():
        if t.isnumeric():
            if int(t) not in index_mapping:
                index_mapping[int(t)] = current_idx
                current_idx += 1
    decoded_labels_ii = []
    for t in left_lf.split():
        if t.isnumeric():
            decoded_labels_ii += [str(index_mapping[int(t)])]
        else:
            decoded_labels_ii += [t]

    index_mapping = {}
    current_idx = 0
    for t in right_lf.split():
        if t.isnumeric():
            if int(t) not in index_mapping:
                index_mapping[int(t)] = current_idx
                current_idx += 1
    decoded_preds_ii = []
    for t in right_lf.split():
        if t.isnumeric():
            decoded_preds_ii += [str(index_mapping[int(t)])]
        else:
            decoded_preds_ii += [t]


    decoded_labels_ii_str = " ".join(decoded_labels_ii)
    decoded_preds_ii_str = " ".join(decoded_preds_ii)

    if decoded_preds_ii_str == decoded_labels_ii_str:
        return True
    return False

recogs_neoD_np_re = re.compile(r"""
    ^
    \s*(\*)?
    \s*(\w+?)\s*
    \(
    \s*(.+?)\s*
    \)
    \s*$""", re.VERBOSE)

recogs_neoD_verb_re = re.compile(r"""
    ^
    \s*(\w+?)\s*
    \(
    \s*([0-9]+?)\s*
    \)
    \s*$""", re.VERBOSE)

recogs_neoD_pred_re = re.compile(r"""
    ^
    \s*(\w+?)\s*
    \(
    \s*(.+?)\s*
    ,
    \s*(.+?)\s*
    \)
    \s*$""", re.VERBOSE)

recogs_neoD_mod_re = re.compile(r"""
    ^
    \s*(\w+?)\s*
    \.
    \s*(\w+?)\s*
    \(
    \s*(.+?)\s*
    ,
    \s*(.+?)\s*
    \)
    \s*$""", re.VERBOSE)

def translate_invariant_form_neoD(lf):
    nouns = lf.split(" AND ")[0].split(" ; ")[:-1]
    complements = set(lf.split(" ; ")[-1].split())
    nouns_map = {}
    new_var = 0
    for noun in nouns:
        # check format.
        if not recogs_neoD_np_re.search(noun):
            return {} # this is format error, we cascade the error.
        _, _, original_var = recogs_neoD_np_re.search(noun).groups()
        if original_var not in complements:
            return {} # var must be used, we cascade the error.
        new_noun = noun.replace(str(original_var), str(new_var))
        nouns_map[original_var] = new_noun
        new_var += 1
        
    nmod_conjs_set = set([])
    conjs = lf.split(" ; ")[-1].split(" AND ")
    vp_conjs_map = {}
    nested_conjs = []
    childen_count_map = {}
    for conj in conjs:
        if "nmod" in conj:
            if not recogs_neoD_mod_re.search(conj):
                return {} # this is format error, we cascade the error.
            role, pred, first_arg, second_arg = recogs_neoD_mod_re.search(conj).groups()
            new_conj = f"{role} . {pred} ( {nouns_map[first_arg]} , {nouns_map[second_arg]} )"
            nmod_conjs_set.add(new_conj)
        else:
            if recogs_neoD_verb_re.search(conj):
                # candidate for mapping verb.
                pred, arg = recogs_neoD_verb_re.search(conj).groups()
                if not arg.isnumeric():
                    return {}
                new_conj = f"{pred}"
                if arg in vp_conjs_map:
                    vp_conjs_map[arg].append(new_conj)
                else:
                    vp_conjs_map[arg] = [new_conj]
                continue
            if not recogs_neoD_pred_re.search(conj):
                return {} # this is format error, we cascade the error.
            
            role, first_arg, second_arg = recogs_neoD_pred_re.search(conj).groups()
            if first_arg == second_arg or first_arg in nouns_map or not first_arg.isnumeric():
                return {} # this is index collision, we cascade the error.
            if second_arg.isnumeric() and second_arg in nouns_map:
                second_arg = nouns_map[second_arg]
                new_conj = f"{role} ( {second_arg} )"
                if first_arg in vp_conjs_map:
                    vp_conjs_map[first_arg].append(new_conj)
                else:
                    vp_conjs_map[first_arg] = [new_conj]
            elif second_arg.isnumeric():
                if first_arg not in childen_count_map:
                    childen_count_map[first_arg] = 1
                else:
                    childen_count_map[first_arg] += 1
                nested_conjs.append({
                    "role": role,
                    "first_arg": first_arg,
                    "second_arg": second_arg,
                })
            else:
                return {}
    
    while_loop_count = 0
    while len(nested_conjs) > 0:
        while_loop_count += 1
        if while_loop_count > 100:
            return {}
        conj = nested_conjs.pop(0)
        if conj['second_arg'] not in childen_count_map or childen_count_map[conj['second_arg']] == 0:
            core = " AND ".join(vp_conjs_map[conj['second_arg']])
            vp_conjs_map[conj['first_arg']].append(f"{conj['role']} ( {core} )")
            childen_count_map[conj['first_arg']] -= 1
        else:
            # if the conj is corrupted, then we abandon just let it go and fail to compare.
            if conj['first_arg'] == conj['second_arg']:
                return {}
            nested_conjs.append(conj)
    
    filtered_conjs_set = set([])
    for k, v in vp_conjs_map.items():
        vp_conjs_map[k].sort()
    for k, v in vp_conjs_map.items():
        vp_expression = " AND ".join(v)
        if vp_expression in filtered_conjs_set:
            return {} # this is not allowed. exact same VP expression is not allowed this time.
        filtered_conjs_set.add(vp_expression)
    for conj in nmod_conjs_set:
        if conj in filtered_conjs_set:
            return {} # this is not allowed. exact same VP expression is not allowed this time.
        filtered_conjs_set.add(conj)
    return filtered_conjs_set

def check_set_equal_neoD(left_lf, right_lf):
    try:
        if translate_invariant_form_neoD(left_lf) == \
        translate_invariant_form_neoD(right_lf):
            return True
        else:
            return False
    except:
        return False