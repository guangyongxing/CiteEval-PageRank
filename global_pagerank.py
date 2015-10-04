#################################################################
#
#   __author__ = 'yanhe'
#
#   global_pagerank:
#       compute the Global PageRank values
#
#################################################################


import scipy.sparse as sparse
import numpy as np
import scipy.spatial.distance as distance


#################################################################
#
#   function matrix_transfer():
#       transition matrix transfer
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
#   function gpr():
#       calculate the global PageRank
#
#################################################################

def gpr():
    # set the value of alpha
    alpha = 0.5
    # get the transition matrix
    trans_mtx = matrix_transfer()
    [row, col] = trans_mtx.shape
    # get the p0 matrix
    p0_mtx = np.divide(np.ones(row), row)

    # initialize the pagerank vector pr_mtx
    pr_mtx = np.random.dirichlet(np.ones(row), size=1).ravel()
    # iteration to update the pr_mtx
    num_of_round = 1
    while num_of_round < 500:
        # print num_of_round
        num_of_round += 1
        pr_mtx_update = (1 - alpha) * (trans_mtx * pr_mtx) + alpha * p0_mtx
        if distance.euclidean(pr_mtx, pr_mtx_update) < pow(10, -13):
            break
        pr_mtx = pr_mtx_update
    file_writer(pr_mtx)
    print '\n' + "PageRank calculation finished." + '\n'
    return pr_mtx


#################################################################
#
#   function file_writer(pr_mtx):
#       write the result into file
#
#################################################################

def file_writer(pr_mtx):
    # write the global pagerank result into txt file
    f = open('global_pagerank_result', 'w')
    for ele in pr_mtx:
        f.write(str(ele) + '\n')


#################################################################
#
#   function main():
#       main function of the program
#
#################################################################

def main():
    pr_mtx = gpr()
    print pr_mtx[0: 10]


# use this line to execute the main function
if __name__ == "__main__":
    main()


# end of the pagerank computation process
