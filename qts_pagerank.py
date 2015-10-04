#################################################################
#
#   __author__ = 'yanhe'
#
#   qts_pagerank:
#       compute the Query-based Topic Sensitive PageRank values
#
#################################################################


import scipy.sparse as sparse
import numpy as np
import scipy.spatial.distance as distance


#################################################################
#
#   function matrix_transfer():
#       get the transition matrix M
#
#################################################################

def matrix_transfer():
    trans_txt_path = "hw3-resources/transition.txt"
    trans_txt = open(trans_txt_path, 'r')
    row_list = []
    col_list = []
    data_list = []
    outer_count = {}
    for line in trans_txt:
        ele_tuple = line.split(' ')
        row_list.append(int(ele_tuple[0]) - 1)
        col_list.append(int(ele_tuple[1]) - 1)
        count = outer_count.get(int(ele_tuple[0]) - 1, 0)
        count += 1
        outer_count[int(ele_tuple[0]) - 1] = count
    size = max(max(row_list), max(col_list)) + 1
    for idx in row_list:
        data_list.append(1.0 / outer_count[idx])
    trans_coo_mtx = sparse.coo_matrix((data_list, (row_list, col_list)), shape=(size, size), dtype=np.float)
    print '\n' + "Transition matrix transfer finished." + '\n'
    # trans_mtx has been transposed
    trans_mtx = trans_coo_mtx.tocsr().transpose()
    return trans_mtx


#################################################################
#
#   function vector_transfer():
#       construct topic-specific teleportation vector p_t
#
#################################################################

def vector_transfer():
    topic_txt_path = "hw3-resources/doc-topics.txt"
    topic_txt = open(topic_txt_path, 'r')
    row_list = []
    col_list = []
    data_list = []
    for line in topic_txt:
        ele_tuple = line.split(' ')
        row_list.append(int(ele_tuple[1]) - 1)
        col_list.append(int(ele_tuple[0]) - 1)
        data_list.append(1)
    row_size = max(row_list) + 1
    col_size = max(col_list) + 1
    topic_coo_mtx = sparse.coo_matrix((data_list, (row_list, col_list)), shape=(row_size, col_size), dtype=np.float)
    # count_in_topic = topic_coo_mtx.sum(axis=1)
    topic_tele_mtx = topic_coo_mtx.toarray() / topic_coo_mtx.toarray().sum(axis=1, keepdims=True)
    return topic_tele_mtx


#################################################################
#
#   function main():
#       main function of the program
#
#################################################################

def main():
    vector_transfer()


# use this line to execute the main function
if __name__ == "__main__":
    main()


# end of the pagerank computation process
