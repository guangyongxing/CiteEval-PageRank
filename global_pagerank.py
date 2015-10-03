#################################################################
#
#   __author__ = 'Yan'
#
#   global_pagerank: compute the GPR values
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
    for line in trans_txt:
        ele_tuple = line.split(' ')
        row_list.append(int(ele_tuple[0]))
        col_list.append(int(ele_tuple[1]))
        data_list.append(int(ele_tuple[2]))
    size = max(max(row_list), max(col_list)) + 1
    trans_coo_mtx = sparse.coo_matrix((data_list, (row_list, col_list)), shape=(size, size), dtype=np.float)
    print '\n' + "Transition matrix transfer finished." + '\n'
    # trans_mtx has been transposed
    trans_mtx = trans_coo_mtx.tocsr().transpose()
    return trans_mtx


#################################################################
#
#   function gpr(): calculate the global PageRank
#
#################################################################
def gpr():
    # get the transition matrix
    trans_mtx = matrix_transfer()
    [row, col] = trans_mtx.shape
    # get the p0 matrix
    p0_mtx = np.transpose(np.empty((1, col)))
    p0_val = 1.0 / row
    p0_mtx.fill(p0_val)
    # set the value of alpha
    alpha = 0.1
    # initialize the pagerank vector pr_mtx
    pr_mtx = np.transpose(np.random.dirichlet(np.ones(row), size=1))
    # iteration 10 rounds to update the pr_mtx
    num_of_round = 0
    while num_of_round < 100:
        print num_of_round
        num_of_round += 1
        pr_mtx_update = np.multiply(1 - alpha, trans_mtx.dot(pr_mtx)) + np.multiply(alpha, p0_mtx)
        print distance.euclidean(pr_mtx, pr_mtx_update)
        if distance.euclidean(pr_mtx, pr_mtx_update) < pow(10, -13):
            break
        pr_mtx = pr_mtx_update

    print '\n' + "PageRank calculation finished." + '\n'


#################################################################
#
#   function main(): main function of the program
#
#################################################################
def main():
    # print '\n' + "start to reading the docVectors data." + '\n'
    gpr()


# use this line to execute the main function
if __name__ == "__main__":
    main()


# end of the pagerank computation process
