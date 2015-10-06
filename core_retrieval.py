#################################################################
#
#   __author__ = 'yanhe'
#
#   core_retrieval:
#       run each retrieval method in a whole process
#
#################################################################

import timeit
import ns_retrieval
import ws_retrieval
import cm_retrieval


#################################################################
#
#   function file_scanner():
#       get the name of provided indri files
#
#################################################################
def core_process():
    # no-search method
    start_ns_gpr = timeit.default_timer()
    ns_retrieval.ns_gpr()
    end_ns_gpr = timeit.default_timer()
    print "***** ns_gpr: " + str(end_ns_gpr - start_ns_gpr) + "*****\n"

    start_ns_qtspr = timeit.default_timer()
    ns_retrieval.ns_qtspr()
    end_ns_qtspr = timeit.default_timer()
    print "***** ns_qtspr: " + str(end_ns_qtspr - start_ns_qtspr) + "*****\n"

    start_ns_ptspr = timeit.default_timer()
    ns_retrieval.ns_ptspr()
    end_ns_ptspr = timeit.default_timer()
    print "***** ns_ptspr: " + str(end_ns_ptspr - start_ns_ptspr) + "*****\n"

    # weighted sum method
    start_ws_gpr = timeit.default_timer()
    ws_retrieval.ws_gpr()
    end_ws_gpr = timeit.default_timer()
    print "***** ws_gpr: " + str(end_ws_gpr - start_ws_gpr) + "*****\n"

    start_ws_qtspr = timeit.default_timer()
    ws_retrieval.ws_qtspr()
    end_ws_qtspr = timeit.default_timer()
    print "***** ws_qtspr: " + str(end_ws_qtspr - start_ws_qtspr) + "*****\n"

    start_ws_ptspr = timeit.default_timer()
    ws_retrieval.ws_ptspr()
    end_ws_ptspr = timeit.default_timer()
    print "***** ws_ptspr: " + str(end_ws_ptspr - start_ws_ptspr) + "*****\n"

    # custom method
    start_cm_gpr = timeit.default_timer()
    cm_retrieval.cm_gpr()
    end_cm_gpr = timeit.default_timer()
    print "***** cm_gpr: " + str(end_cm_gpr - start_cm_gpr) + "*****\n"

    start_cm_qtspr = timeit.default_timer()
    cm_retrieval.cm_qtspr()
    end_cm_qtspr = timeit.default_timer()
    print "***** cm_qtspr: " + str(end_cm_qtspr - start_cm_qtspr) + "*****\n"

    start_cm_ptspr = timeit.default_timer()
    cm_retrieval.cm_ptspr()
    end_cm_ptspr = timeit.default_timer()
    print "***** cm_ptspr: " + str(end_cm_ptspr - start_cm_ptspr) + "*****\n"


# use this line to execute the main function
if __name__ == "__main__":
    print "Starting the retrieval process." + '\n'
    core_process()


# end of the process
