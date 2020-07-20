import logging
import json
import os
from deepkp.utils.misc import normalize_string
from collections import OrderedDict
from prettytable import PrettyTable
from collections import defaultdict
from deepkp.eval.kpeval.util import *
from tqdm import tqdm

logger = logging.getLogger()


def log_result(score_dict, time, mode, total_example):
    """
    :param result:
    :param time: total time used for evaluation
    :param mode: train, valid or test
    :return:
    """
    # score_dict['topk_dict'] = {'present': [5, 'M'], 'absent': [5, 'M'], 'all': [5, 'M']}
    result = dict()
    topk_dict = score_dict['topk_dict']
    for tag, topk_list in topk_dict.items():
        tag = tag.lower()
        for topk in topk_list:
            # Compute the macro averaged recall, precision and F-1 score
            macro_avg_precision_k = sum(score_dict['precision@{}_{}'.format(topk, tag)]) / len(
                score_dict['precision@{}_{}'.format(topk, tag)])
            macro_avg_recall_k = sum(score_dict['recall@{}_{}'.format(topk, tag)]) / len(
                score_dict['recall@{}_{}'.format(topk, tag)])
            macro_avg_f1_score_k = (2 * macro_avg_precision_k * macro_avg_recall_k) / \
                                   (macro_avg_precision_k + macro_avg_recall_k) if \
                (macro_avg_precision_k + macro_avg_recall_k) > 0 else 0.0
            _topk = topk.lower() if isinstance(topk, str) else topk
            result['precision_at_{}_{}'.format(_topk, tag)] = round(macro_avg_precision_k, 3)
            result['recall_at_{}_{}'.format(_topk, tag)] = round(macro_avg_recall_k, 3)
            result['f1_at_{}_{}'.format(_topk, tag)] = round(macro_avg_f1_score_k, 3)

    log = '%s evaluation: examples = %d, elapsed time = %.2f' % \
          (mode, total_example, time)
    logger.info(log)

    table = PrettyTable()
    table.field_names = ["Metric Name", "Present", "Absent", "All"]
    table.align["Metric Name"] = "l"
    table.align["Present"] = "r"
    table.align["Absent"] = "r"
    table.align["Overall"] = "r"

    for metric in ["precision", "recall", "f1"]:
        for k in topk_dict["present"]:
            k = k.lower() if isinstance(k, str) else k
            row = ['{}_at_{}'.format(metric, k)]
            for tag in ["present", "absent", "all"]:
                row += [result['{}_at_{}_{}'.format(metric, k, tag)]]
            table.add_row(row)

    logger.info('\n' + table.get_string(title="Evaluation results"))
    return result


def eval_accuracies(present_absent_kps, filename=None,
                    disable_extra_one_word_filter=False, k_list=[5, 'M']):
    """An official evalutation helper.
     Arguments:
        present_absent_kps: a list of dict {'present': [], 'absent': []}.
        filename:
    """
    score_dict = defaultdict(list)
    k_list = process_input_ks(k_list)
    topk_dict = {'present': k_list, 'absent': k_list, 'all': k_list}

    eval_output = []
    for i in range(len(present_absent_kps)):
        pred_pkp = present_absent_kps[i]['present']['pred']
        pred_akp = present_absent_kps[i]['absent']['pred']
        gold_pkp = present_absent_kps[i]['present']['gold']
        gold_akp = present_absent_kps[i]['absent']['gold']
        stemmed_pred_pkp = [normalize_string(nh[0], dostem=True) for nh in pred_pkp]
        stemmed_pred_akp = [normalize_string(nh[0], dostem=True) for nh in pred_akp]
        dup_removed = set()
        printable_pkp, printable_akp = [], []
        for nh, sh in zip(pred_pkp, stemmed_pred_pkp):
            if sh not in dup_removed:
                dup_removed.add(sh)
                printable_pkp.append(nh)
        for nh, sh in zip(pred_akp, stemmed_pred_akp):
            if sh not in dup_removed:
                dup_removed.add(sh)
                printable_akp.append(nh)
        # calculate score
        score_dict = evaluate([i[0] for i in printable_pkp],
                              [i[0] for i in printable_akp],
                              gold_pkp, gold_akp,
                              score_dict, topk_dict, disable_extra_one_word_filter)
        logobj = OrderedDict()
        if 'id' in present_absent_kps[i]:
            logobj['id'] = present_absent_kps[i]['id']
        logobj['keyphrases'] = {'present': printable_pkp, 'absent': printable_akp}
        logobj['references'] = {'present': gold_pkp, 'absent': gold_akp}
        key = 'f1_score@%s_present' % str(k_list[0])
        logobj[key] = score_dict[key][i]
        eval_output.append(logobj)

    if filename:
        with open(filename, 'w') as fw:
            json.dump(eval_output, fw, indent=4)

    score_dict['topk_dict'] = topk_dict
    return score_dict


def update_score_dict(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed, k_list, score_dict, tag):
    num_targets = len(trg_token_2dlist_stemmed)
    num_predictions = len(pred_token_2dlist_stemmed)

    is_match = compute_match_result(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed,
                                    type='exact', dimension=1)
    # Classification metrics
    precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks = \
        compute_classification_metrics_at_ks(is_match, num_predictions, num_targets, k_list=k_list,
                                             meng_rui_precision=False)

    for topk, precision_k, recall_k, f1_k, num_matches_k, num_predictions_k in \
            zip(k_list, precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks):
        score_dict['precision@{}_{}'.format(topk, tag)].append(precision_k)
        score_dict['recall@{}_{}'.format(topk, tag)].append(recall_k)
        score_dict['f1_score@{}_{}'.format(topk, tag)].append(f1_k)
        score_dict['num_matches@{}_{}'.format(topk, tag)].append(num_matches_k)
        score_dict['num_predictions@{}_{}'.format(topk, tag)].append(num_predictions_k)
        score_dict['num_targets@{}_{}'.format(topk, tag)].append(num_targets)

    score_dict['num_targets_{}'.format(tag)].append(num_targets)
    score_dict['num_predictions_{}'.format(tag)].append(num_predictions)
    return score_dict


def evaluate(pred_pkp_list, pred_akp_list, trg_pkp_list, trg_akp_list,
             score_dict, topk_dict, disable_extra_one_word_filter=False):
    # convert the str to token list
    pkp_pred_token_2dlist = [pred_str.strip().split(' ') for pred_str in pred_pkp_list]
    akp_pred_token_2dlist = [pred_str.strip().split(' ') for pred_str in pred_akp_list]
    pkp_token_2dlist = [trg_str.strip().split(' ') for trg_str in trg_pkp_list]
    akp_token_2dlist = [trg_str.strip().split(' ') for trg_str in trg_akp_list]

    # perform stemming
    pkp_stemmed_pred_token_2dlist = stem_str_list(pkp_pred_token_2dlist)
    akp_stemmed_pred_token_2dlist = stem_str_list(akp_pred_token_2dlist)
    pkp_stemmed_trg_token_2dlist = stem_str_list(pkp_token_2dlist)
    akp_stemmed_trg_token_2dlist = stem_str_list(akp_token_2dlist)

    # Filter out duplicate, invalid, and extra one word predictions
    present_filtered_stemmed_pred_token_2dlist, _ = filter_prediction(True,
                                                                      disable_extra_one_word_filter,
                                                                      pkp_stemmed_pred_token_2dlist)
    absent_filtered_stemmed_pred_token_2dlist, _ = filter_prediction(True,
                                                                     disable_extra_one_word_filter,
                                                                     akp_stemmed_pred_token_2dlist)

    # Remove duplicated targets
    present_unique_stemmed_trg_token_2dlist, _ = find_unique_target(pkp_stemmed_trg_token_2dlist)
    absent_unique_stemmed_trg_token_2dlist, _ = find_unique_target(akp_stemmed_trg_token_2dlist)

    unique_stemmed_trg_token_2dlist = present_unique_stemmed_trg_token_2dlist + \
                                      absent_unique_stemmed_trg_token_2dlist
    filtered_stemmed_pred_token_2dlist = present_filtered_stemmed_pred_token_2dlist + \
                                         absent_filtered_stemmed_pred_token_2dlist

    # compute all the metrics and update the score_dict
    score_dict = update_score_dict(unique_stemmed_trg_token_2dlist,
                                   filtered_stemmed_pred_token_2dlist,
                                   topk_dict['all'], score_dict, 'all')
    # compute all the metrics and update the score_dict for present keyphrase
    score_dict = update_score_dict(present_unique_stemmed_trg_token_2dlist,
                                   present_filtered_stemmed_pred_token_2dlist,
                                   topk_dict['present'], score_dict, 'present')
    # compute all the metrics and update the score_dict for absent keyphrase
    score_dict = update_score_dict(absent_unique_stemmed_trg_token_2dlist,
                                   absent_filtered_stemmed_pred_token_2dlist,
                                   topk_dict['absent'], score_dict, 'absent')

    return score_dict


def main(predictions, exp_path, result_file_suffix, k_list=[5, 'M']):
    score_dict = defaultdict(list)
    k_list = process_input_ks(k_list)
    topk_dict = {'present': k_list, 'absent': k_list, 'all': k_list}

    total_num_src = 0
    total_num_src_with_present_keyphrases = 0
    total_num_src_with_absent_keyphrases = 0
    total_num_unique_predictions = 0
    total_num_present_filtered_predictions = 0
    total_num_present_unique_targets = 0
    total_num_absent_filtered_predictions = 0
    total_num_absent_unique_targets = 0
    max_unique_targets = 0

    sum_incorrect_fraction_for_identifying_present = 0
    sum_incorrect_fraction_for_identifying_absent = 0

    for data_idx, item in enumerate(tqdm(predictions, total=len(predictions))):
        total_num_src += 1
        pred_pkp = item['present']['pred']
        pred_akp = item['absent']['pred']
        if pred_pkp and isinstance(pred_pkp[0], tuple):
            pred_pkp = [i[0] for i in pred_pkp]
        if pred_akp and isinstance(pred_akp[0], tuple):
            pred_akp = [i[0] for i in pred_akp]

        pred_pkp_2dlist = [pred_str.strip().split(' ') for pred_str in pred_pkp]
        pred_akp_2dlist = [pred_str.strip().split(' ') for pred_str in pred_akp]
        trg_pkp_2dlist = [pred_str.strip().split(' ') for pred_str in item['present']['gold']]
        trg_akp_2dlist = [pred_str.strip().split(' ') for pred_str in item['absent']['gold']]

        num_predictions = len(pred_pkp_2dlist) + len(pred_akp_2dlist)

        # perform stemming
        stemmed_trg_pkp_2dlist = stem_str_list(trg_pkp_2dlist)
        stemmed_trg_akp_2dlist = stem_str_list(trg_akp_2dlist)
        stemmed_pred_pkp_2dlist = stem_str_list(pred_pkp_2dlist)
        stemmed_pred_akp_2dlist = stem_str_list(pred_akp_2dlist)

        # Filter out duplicate, invalid, and extra one word predictions
        filtered_stemmed_pred_pkp_2dlist, num_duplicated_pkp_predictions = \
            filter_prediction(INVALIDATE_UNK, DISABLE_EXTRA_ONE_WORD_FILTER, stemmed_pred_pkp_2dlist)
        filtered_stemmed_pred_akp_2dlist, num_duplicated_akp_predictions = \
            filter_prediction(INVALIDATE_UNK, DISABLE_EXTRA_ONE_WORD_FILTER, stemmed_pred_akp_2dlist)

        filtered_stemmed_pred_kp_2dlist = filtered_stemmed_pred_pkp_2dlist + filtered_stemmed_pred_akp_2dlist

        num_duplicated_predictions = num_duplicated_pkp_predictions + num_duplicated_akp_predictions
        total_num_unique_predictions += (num_predictions - num_duplicated_predictions)
        num_filtered_predictions = len(filtered_stemmed_pred_pkp_2dlist) + len(filtered_stemmed_pred_akp_2dlist)

        # Remove duplicated targets
        unique_stemmed_trg_pkp_2dlist, num_duplicated_pkp_trg = find_unique_target(stemmed_trg_pkp_2dlist)
        unique_stemmed_trg_akp_2dlist, num_duplicated_akp_trg = find_unique_target(stemmed_trg_akp_2dlist)
        unique_stemmed_trg_kp_2dlist = unique_stemmed_trg_pkp_2dlist + unique_stemmed_trg_akp_2dlist

        # unique_stemmed_trg_token_2dlist = stemmed_trg_token_2dlist
        num_unique_targets = len(unique_stemmed_trg_pkp_2dlist) + len(unique_stemmed_trg_akp_2dlist)

        if num_unique_targets > max_unique_targets:
            max_unique_targets = num_unique_targets

        total_num_present_filtered_predictions += len(filtered_stemmed_pred_pkp_2dlist)
        total_num_present_unique_targets += len(unique_stemmed_trg_pkp_2dlist)
        total_num_absent_filtered_predictions += len(filtered_stemmed_pred_akp_2dlist)
        total_num_absent_unique_targets += len(unique_stemmed_trg_akp_2dlist)
        if len(unique_stemmed_trg_pkp_2dlist) > 0:
            total_num_src_with_present_keyphrases += 1
        if len(unique_stemmed_trg_akp_2dlist) > 0:
            total_num_src_with_absent_keyphrases += 1

        # compute all the metrics and update the score_dict
        score_dict = update_score_dict(unique_stemmed_trg_kp_2dlist, filtered_stemmed_pred_kp_2dlist,
                                       topk_dict['all'], score_dict, 'all')
        # compute all the metrics and update the score_dict for present keyphrase
        score_dict = update_score_dict(unique_stemmed_trg_pkp_2dlist,
                                       filtered_stemmed_pred_pkp_2dlist,
                                       topk_dict['present'], score_dict, 'present')
        # compute all the metrics and update the score_dict for absent keyphrase
        score_dict = update_score_dict(unique_stemmed_trg_akp_2dlist,
                                       filtered_stemmed_pred_akp_2dlist,
                                       topk_dict['absent'], score_dict, 'absent')

    total_num_unique_targets = total_num_present_unique_targets + total_num_absent_unique_targets
    total_num_filtered_predictions = total_num_present_filtered_predictions + total_num_absent_filtered_predictions

    result_txt_str = ""

    # report global statistics
    result_txt_str += (
            'Total #samples: %d\t # samples with present keyphrases: %d\t # samples with absent keyphrases: %d\n' % (
        total_num_src, total_num_src_with_present_keyphrases, total_num_src_with_absent_keyphrases))
    result_txt_str += ('Max. unique targets per src: %d\n' % (max_unique_targets))
    result_txt_str += ('Total #unique predictions: %d\n' % total_num_unique_predictions)

    # report statistics and scores for all predictions and targets
    result_txt_str_all, field_list_all, result_list_all = report_stat_and_scores(total_num_filtered_predictions,
                                                                                 total_num_unique_targets,
                                                                                 total_num_src, score_dict,
                                                                                 topk_dict['all'], 'all')
    result_txt_str_present, field_list_present, result_list_present = report_stat_and_scores(
        total_num_present_filtered_predictions, total_num_present_unique_targets, total_num_src, score_dict,
        topk_dict['present'], 'present')
    result_txt_str_absent, field_list_absent, result_list_absent = report_stat_and_scores(
        total_num_absent_filtered_predictions, total_num_absent_unique_targets, total_num_src, score_dict,
        topk_dict['absent'], 'absent')
    result_txt_str += (result_txt_str_all + result_txt_str_present + result_txt_str_absent)
    field_list = field_list_all + field_list_present + field_list_absent
    result_list = result_list_all + result_list_present + result_list_absent

    # Write to files
    # topk_dict = {'present': [5, 10, 'M'], 'absent': [5, 10, 50, 'M'], 'all': [5, 10, 'M']}
    results_txt_file = open(os.path.join(exp_path, "results_log_{}.txt".format(result_file_suffix)), "w")

    result_txt_str += "===================================Separation====================================\n"
    result_txt_str += "Avg error fraction for identifying present keyphrases: {:.5}\n".format(
        sum_incorrect_fraction_for_identifying_present / total_num_src)
    result_txt_str += "Avg error fraction for identifying absent keyphrases: {:.5}\n".format(
        sum_incorrect_fraction_for_identifying_absent / total_num_src)

    # Report MAE on lengths
    result_txt_str += "===================================MAE stat====================================\n"

    num_targets_present_array = np.array(score_dict['num_targets_present'])
    num_predictions_present_array = np.array(score_dict['num_predictions_present'])
    num_targets_absent_array = np.array(score_dict['num_targets_absent'])
    num_predictions_absent_array = np.array(score_dict['num_predictions_absent'])

    all_mae = mae(num_targets_present_array + num_targets_absent_array,
                  num_predictions_present_array + num_predictions_absent_array)
    present_mae = mae(num_targets_present_array, num_predictions_present_array)
    absent_mae = mae(num_targets_absent_array, num_predictions_absent_array)

    result_txt_str += "MAE on keyphrase numbers (all): {:.5}\n".format(all_mae)
    result_txt_str += "MAE on keyphrase numbers (present): {:.5}\n".format(present_mae)
    result_txt_str += "MAE on keyphrase numbers (absent): {:.5}\n".format(absent_mae)

    results_txt_file.write(result_txt_str)
    results_txt_file.close()

    return


def run_eval(predictions, dir_name, file_suffix, disable_extra_one_word_filter, k_list):
    global DISABLE_EXTRA_ONE_WORD_FILTER
    DISABLE_EXTRA_ONE_WORD_FILTER = disable_extra_one_word_filter
    main(predictions, dir_name, file_suffix, k_list)
