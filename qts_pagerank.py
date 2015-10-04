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
#   function vector_transfer():
#       construct topic-specific teleportation vector p_t
#
#################################################################

def vector_transfer():
    topic_txt_path = "hw3-resources/doc-topics.txt"
    topic_txt = open(topic_txt_path, 'r')
    topic_count = {}
    # for line in topic_txt:
    #     ele_tuple = line.split(' ')
    #     count = topic_count.get(int(ele_tuple[1]), 0)
    #     count += 1
    #     topic_count[int(ele_tuple[1])] = count
    # tstele_vec = []
    # for idx in range(1, len(topic_count)):
    #     tstele_vec.append(1.0 / topic_count[idx])


#################################################################
#
#   function main():
#       main function of the program
#
#################################################################

def main():
    print "hello"


# use this line to execute the main function
if __name__ == "__main__":
    main()


# end of the pagerank computation process
