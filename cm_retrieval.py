#################################################################
#
#   __author__ = 'yanhe'
#
#   cm_retrieval:
#       re-rank the document with custom retrieval method
#
#################################################################


import glob
import global_pagerank
import qts_pagerank
import pts_pagerank
import numpy as np
from operator import add
import math


#################################################################
#
#   function file_scanner():
#       get the name of provided indri files
#
#################################################################
def file_scanner():
    # print "Reading Indri files." + '\n'
    indri_path = "hw3-resources/indri-lists/*.txt"
    indri_files = glob.glob(indri_path)
    indri_file_names = {}
    for cur_file_name in indri_files:
        # query_id is the user-query pair in the file name
        query_id = cur_file_name.split('/')[2].split('.')[0]
        cur_num = int(query_id.split('-')[0] + query_id.split('-')[1])
        indri_file_names[cur_num] = [query_id, cur_file_name]
    return indri_file_names


#################################################################
#
#   function docid_extracter():
#       get the doc_id in the indri file
#
#################################################################

def doc_extracter(path):
    cur_file = open(path, 'r')
    doc_id = []
    for line in cur_file:
        doc_id.append(int(line.split(' ')[2]) - 1)
    return doc_id


#################################################################
#
#   function docid_extracter():
#       get the doc_id in the indri file
#       TODO: refine
#
#################################################################
def score_extracter(path):
    cur_file = open(path, 'r')
    indri_score = []
    for line in cur_file:
        indri_score.append(float(line.split(' ')[4]))
    return indri_score


#################################################################
#
#   function cosine_inter():
#       custom method
#
#################################################################

def cosine_inter(mu_list):
    return [(1 - math.cos(mu * math.pi)) / 2 for mu in mu_list]


#################################################################
#
#   function ws_gpr():
#       compute weighted sum global pagerank ranking
#
#################################################################
def cm_gpr():
    # get the global pagerank result
    gpr_mtx = global_pagerank.gpr()
    # get the indri file names
    indri_names = file_scanner()
    # write the ranking result into txt file
    f = open('rank/cm_gpr_rank.txt', 'w')
    for cur_num in sorted(indri_names):
        query_id = indri_names[cur_num][0]
        file_name = indri_names[cur_num][1]
        # doc id in the current indri file
        doc_id = doc_extracter(file_name)
        doc_num = len(doc_id)
        # normalize intri score for each doc
        indri_score = score_extracter(file_name)
        # indri_score_pos = np.subtract(indri_score, min(indri_score) - 1)
        indri_score_pos = np.power(math.e, indri_score)
        # transform to all positive value
        indri_norm = [float(i)/sum(indri_score_pos) for i in indri_score_pos]
        # normalize pagerank value
        gpr_value = gpr_mtx[doc_id]
        gpr_norm = [float(i)/sum(gpr_value) for i in gpr_value]
        mu_list = np.arange(0.85, 0.95, 0.1 / doc_num)[::-1]
        decay_list = np.arange(0.65, 1.0, 0.35 / doc_num)
        mu2_list = cosine_inter(mu_list)
        mu3_list = np.multiply(np.subtract(1.0, mu2_list), decay_list)
        # ws_score = map(add, np.multiply(indri_norm, mu2_list), np.multiply(gpr_norm, np.subtract(1.0, mu2_list)))
        ws_score = np.subtract(np.multiply(indri_norm, mu2_list), np.multiply(gpr_norm, mu3_list))
        # sort by descending order
        gpr_score = np.argsort(ws_score)[::-1].tolist()
        doc_id_arr = np.array(doc_id)
        gpr_rank = doc_id_arr[gpr_score]
        rank_num = 0
        for idx in gpr_rank:
            rank_num += 1
            f.write("{} Q0 {} {} {} run-1\n".format(query_id, idx + 1, rank_num, ws_score[doc_id.index(idx)]))
    f.close()
    print "Custom method GPR ranking finished." + '\n'


#################################################################
#
#   function ws_qtspr():
#       compute weighted sum query-based TSPR ranking
#
#################################################################

def cm_qtspr():
    # get the query-based pagerank result
    qtspr_mtx = qts_pagerank.online_tspr()
    # get the indri file names
    indri_names = file_scanner()
    # write the ranking result into txt file
    f = open('rank/cm_qtspr_rank.txt', 'w')
    query_count = -1
    for cur_num in sorted(indri_names):
        query_count += 1
        query_id = indri_names[cur_num][0]
        file_name = indri_names[cur_num][1]
        # doc id in the current indri file
        doc_id = doc_extracter(file_name)
        doc_num = len(doc_id)
        # normalize intri score for each doc
        indri_score = score_extracter(file_name)
        indri_score_pos = np.power(math.e, indri_score)
        # transform to all positive value
        indri_norm = [float(i)/sum(indri_score_pos) for i in indri_score_pos]
        # normalize pagerank value
        qtspr_value = qtspr_mtx[query_count][doc_id]
        qtspr_norm = [float(i)/sum(qtspr_value) for i in qtspr_value]
        # use custom method to combine
        mu_list = np.arange(0.86, 0.96, 0.1 / doc_num)[::-1]
        decay_list = np.arange(0.65, 1.0, 0.35 / doc_num)
        mu2_list = cosine_inter(mu_list)
        mu3_list = np.multiply(np.subtract(1.0, mu2_list), decay_list)
        # TODO: change the pagerank score to be penalty
        # ws_score = map(add, np.multiply(indri_norm, mu2_list), np.multiply(qtspr_norm, np.subtract(1.0, mu2_list)))
        ws_score = np.subtract(np.multiply(indri_norm, mu2_list), np.multiply(qtspr_norm, mu3_list))
        # sort by descending order
        qtspr_score = np.argsort(ws_score)[::-1].tolist()
        doc_id_arr = np.array(doc_id)
        qtspr_rank = doc_id_arr[qtspr_score]
        rank_num = 0
        for idx in qtspr_rank:
            rank_num += 1
            f.write("{} Q0 {} {} {} run-1\n".format(query_id, idx + 1, rank_num, ws_score[doc_id.index(idx)]))
    f.close()
    print "Custom method query-based TSPR ranking finished." + '\n'


#################################################################
#
#   function ws_ptspr():
#       compute weighted sum personalized TSPR ranking
#
#################################################################

def cm_ptspr():
    # get the query-based pagerank result
    ptspr_mtx = pts_pagerank.online_tspr()
    # get the indri file names
    indri_names = file_scanner()
    # write the ranking result into txt file
    f = open('rank/cm_ptspr_rank.txt', 'w')
    query_count = -1
    for cur_num in sorted(indri_names):
        query_count += 1
        query_id = indri_names[cur_num][0]
        file_name = indri_names[cur_num][1]
        # doc id in the current indri file
        doc_id = doc_extracter(file_name)
        doc_num = len(doc_id)
        # normalize intri score for each doc
        indri_score = score_extracter(file_name)
        indri_score_pos = np.power(math.e, indri_score)
        # transform to all positive value
        indri_norm = [float(i)/sum(indri_score_pos) for i in indri_score_pos]
        # normalize pagerank value
        ptspr_value = ptspr_mtx[query_count][doc_id]
        ptspr_norm = [float(i)/sum(ptspr_value) for i in ptspr_value]
        # use custom method to combine
        mu_list = np.arange(0.85, 0.95, 0.1 / doc_num)[::-1]
        decay_list = np.arange(0.45, 1.0, 0.55 / doc_num)
        mu2_list = cosine_inter(mu_list)
        mu3_list = np.multiply(np.subtract(1.0, mu2_list), decay_list)
        # ws_score = map(add, np.multiply(indri_norm, mu2_list), np.multiply(ptspr_norm, np.subtract(1.0, mu2_list)))
        ws_score = np.subtract(np.multiply(indri_norm, mu2_list), np.multiply(ptspr_norm, mu3_list))
        # sort by descending order
        ptspr_score = np.argsort(ws_score)[::-1].tolist()
        doc_id_arr = np.array(doc_id)
        qtspr_rank = doc_id_arr[ptspr_score]
        rank_num = 0
        for idx in qtspr_rank:
            rank_num += 1
            f.write("{} Q0 {} {} {} run-1\n".format(query_id, idx + 1, rank_num, ws_score[doc_id.index(idx)]))
    f.close()
    print "Custom method personalized TSPR ranking finished." + '\n'

# use this line to execute the main function
if __name__ == "__main__":
    print "Starting the custom method for retrieval." + '\n'
    cm_gpr()
    # cm_qtspr()
    # cm_ptspr()


# end of the process
