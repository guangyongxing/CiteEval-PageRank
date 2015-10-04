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
#       get the transition matrix M.T
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
#   function offline_tspr():
#       compute offline TSPR vectors
#
#################################################################
def offline_tspr():
    # set the value of alpha, beta, gamma
    alpha = 0.5
    beta = 0.4
    gamma = 0.1
    trans_mtx = matrix_transfer()
    [row, col] = trans_mtx.shape
    topic_tele_mtx = vector_transfer()
    topic_num = len(topic_tele_mtx)
    tspr_vec = []
    for idx in range(0, topic_num):
        cur_topic_vec = topic_tele_mtx[idx]
        # get the p0 matrix
        p0_mtx = np.divide(np.ones(row), row)
        # initialize the pagerank vector pr_mtx
        cur_pr_mtx = np.random.dirichlet(np.ones(row), size=1).ravel()

        # iteration to update the cur_pr_mtx
        num_of_round = 1
        while num_of_round < 500:
            # print num_of_round
            num_of_round += 1
            cur_pr_mtx_update = alpha * (trans_mtx * cur_pr_mtx) + beta * cur_topic_vec + gamma * p0_mtx
            if distance.euclidean(cur_pr_mtx, cur_pr_mtx_update) < pow(10, -13):
                break
            cur_pr_mtx = cur_pr_mtx_update
        tspr_vec.append(cur_pr_mtx)

    print '\n' + "Offline TSPR matrix generated." + '\n'
    return tspr_vec


#################################################################
#
#   function main():
#       main function of the program
#
#################################################################

def main():
    offline_tspr()


# use this line to execute the main function
if __name__ == "__main__":
    main()


# end of the pagerank computation process
