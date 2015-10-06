#################################################################
#
#   __author__ = 'yanhe'
#
#   ns_retrieval:
#       re-rank the document with no-search retrieval method
#
#################################################################


import glob
import global_pagerank
import qts_pagerank
import pts_pagerank
import numpy as np


#################################################################
#
#   function file_scanner():
#       get the name of provided indri files
#
#################################################################
def file_scanner():
    print "Reading Indri files." + '\n'
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
#   function ns_gpr():
#       compute the no-search global pagerank ranking
#
#################################################################
def ns_gpr():
    # get the global pagerank result
    gpr_mtx = global_pagerank.gpr()
    # get the indri file names
    indri_names = file_scanner()
    # write the ranking result into txt file
    f = open('rank/ns_gpr_rank', 'w')
    for cur_num in sorted(indri_names):
        query_id = indri_names[cur_num][0]
        file_name = indri_names[cur_num][1]
        # doc id in the current indri file
        doc_id = doc_extracter(file_name)
        # sort by descending order
        gpr_score = np.argsort(gpr_mtx[doc_id])[::-1].tolist()
        doc_id_arr = np.array(doc_id)
        gpr_rank = doc_id_arr[gpr_score]
        rank_num = 0
        for idx in gpr_rank:
            rank_num += 1
            f.write("{} Q0 {} {} {} run-1\n".format(query_id, idx + 1, rank_num, gpr_mtx[idx]))
    f.close()
    print "No-search GPR ranking finished." + '\n'


#################################################################
#
#   function ns_qtspr():
#       compute the no-search query-based TSPR ranking
#
#################################################################

def ns_qtspr():
    # get the query-based pagerank result
    qtspr_mtx = qts_pagerank.online_tspr()
    # get the indri file names
    indri_names = file_scanner()
    # write the ranking result into txt file
    f = open('rank/ns_qtspr_rank', 'w')
    query_count = -1
    for cur_num in sorted(indri_names):
        query_count += 1
        query_id = indri_names[cur_num][0]
        file_name = indri_names[cur_num][1]
        # doc id in the current indri file
        doc_id = doc_extracter(file_name)
        # sort by descending order
        qtspr_score = np.argsort(qtspr_mtx[query_count][doc_id])[::-1].tolist()
        doc_id_arr = np.array(doc_id)
        qtspr_rank = doc_id_arr[qtspr_score]
        rank_num = 0
        for idx in qtspr_rank:
            rank_num += 1
            f.write("{} Q0 {} {} {} run-1\n".format(query_id, idx + 1, rank_num, qtspr_mtx[query_count][idx]))
    f.close()
    print "No-search QTSPR ranking finished." + '\n'


#################################################################
#
#   function ns_ptspr():
#       compute the no-search personalized TSPR ranking
#
#################################################################
def ns_ptspr():
    # get the query-based pagerank result
    ptspr_mtx = pts_pagerank.online_tspr()
    # get the indri file names
    indri_names = file_scanner()
    # write the ranking result into txt file
    f = open('rank/ns_ptspr_rank', 'w')
    query_count = -1
    for cur_num in sorted(indri_names):
        query_count += 1
        query_id = indri_names[cur_num][0]
        file_name = indri_names[cur_num][1]
        # doc id in the current indri file
        doc_id = doc_extracter(file_name)
        # sort by descending order
        qtspr_score = np.argsort(ptspr_mtx[query_count][doc_id])[::-1].tolist()
        doc_id_arr = np.array(doc_id)
        qtspr_rank = doc_id_arr[qtspr_score]
        rank_num = 0
        for idx in qtspr_rank:
            rank_num += 1
            f.write("{} Q0 {} {} {} run-1\n".format(query_id, idx + 1, rank_num, ptspr_mtx[query_count][idx]))
    f.close()
    print "No-search QTSPR ranking finished." + '\n'


# use this line to execute the main function
if __name__ == "__main__":
    ns_gpr()
    # ns_ptspr()


# end of the process
