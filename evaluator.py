import pandas as pd
import numpy as np
import os
import json
import itertools
import math
import statistics

FOLDER_PATH = "./new/results/"
TOPXPERCENTAGE_AS_FLOAT = 0.01

path_list= [ 
    './data/FEN1_5fv7_scoring_and_consensus_maxAL.csv', 
    './data/GBA_2v3e_scoring_and_consensus_maxAL.csv', 
    './data/KAT2A_5h84_scoring_and_consensus_maxAL.csv', 
    './data/MAPK1_2ojg_scoring_and_consensus_maxAL.csv', 
    './data/PKM2_3gr4_scoring_and_consensus_maxAL.csv', 
    './data/VDR_3a2j_scoring_and_consensus_maxAL.csv']

_param_combinations_to_avg = {'learner': ['rf_learner'], 
                             'hyperparameter_tuning': [False], 
                             'batch_size_percentage': [1,0.5,0.1,0.01], 
                             'smids_input_path': ['./data/GBA_2v3e_scoring_and_consensus_maxAL.csv'], 
                             'ground_truth_path': ['./data/GBA_2v3e_scoring_and_consensus_maxAL.csv'], 
                             'cycles': [10], 
                             'column_to_learn': [''], 
                             'do_scoring_function_list_prediction': [False], 
                             'first_query_function': ['random_query_function'], 
                             'query_function': ['greedy_query_function']}

keys, values = zip(*_param_combinations_to_avg.items())
_combinations_to_avg = [dict(zip(keys, v)) for v in itertools.product(*values)]

def coordinate_friendly_list_print(lis):
    if len(lis)==2:
        print("("+str(0)+","+str(lis[0])+")")
        print("("+str(10)+","+str(lis[1])+")")
        print("\n")
    else:
        for i in range(0,len(lis)):
            print("("+str(i)+","+str(lis[i])+")")
    

def enrichmentfactor(activity_path, cycle_protocol):
    pass

def is_dict_in_list(target_dict, dict_list):
    #checks whether a dictionary is in a list of dictionaries
    return any(all(item.get(k) == v for k, v in target_dict.items()) for item in dict_list)

def get_smallest(a,b,c):
    return min(a,b,c)

def get_largest(a,b,c):
    return max(a,b,c)

def get_middle(a,b,c):
    return a + b + c - get_smallest(a,b,c) - get_largest(a,b,c)


def topX(ground_truth, cycle_protocol, x=TOPXPERCENTAGE_AS_FLOAT, column=""):
    ground_truth = pd.read_csv(ground_truth)
    
    if x <1:#percentage case
        final_x = math.floor(ground_truth.shape[0]*x)
    else:
        final_x = x

    if column == "":
        column = ground_truth.columns[-1]

    lis = []

    cyc_col=cycle_protocol.columns
    ground_truth.sort_values(by=[column], ascending=False, inplace=True)
    for i in range(1, cycle_protocol.shape[1]):
        cycle_protocol.sort_values(by=[cyc_col[i]], ascending=False, inplace=True, )
        cycle_protocol_head = cycle_protocol.head(final_x)
        ground_truth_head = ground_truth.head(final_x)
        count = 0
        lookup = set(ground_truth_head["ID"])
        for j in cycle_protocol_head["ID"]:
            if j in lookup:
                count += 1
        lis.append((count/final_x)*100)
    return lis


def avg_combinatorial_space_no_wiskers(combinations_to_avg):

    cycles=combinations_to_avg[0]["cycles"]
    avg_counter=0
    if cycles == 10:
        res = [0,0,0,0,0,0,0,0,0,0,0]
    else:
        res= [0,0]
    for hash in os.listdir(FOLDER_PATH):
        combination_path = FOLDER_PATH+ hash + '/combination.txt'
        if not os.path.exists(combination_path):
            #print("no combination for hash", hash)
            continue

        combinations = {}
        with open(FOLDER_PATH+ hash + '/combination.txt', 'r') as file:
            combinations = json.loads(file.read())

        if not is_dict_in_list(combinations, combinations_to_avg):
            continue #skipping hash


        data_path = combinations["ground_truth_path"]
        stat_counter =0
        for stat in os.listdir(FOLDER_PATH+ hash):
            
            if (stat == "combination.txt") | ( not os.path.exists(FOLDER_PATH+ hash + '/' + stat + "/cache.csv")):
                continue # skipping combination file or if no chache is present
            if stat == "7" or stat == "8" or stat == "9":
                continue

            avg_counter +=1
            
            cycle_protocol = pd.read_csv(FOLDER_PATH+ hash + '/' + stat+"/cache.csv")#read cycle protocol


            if (cycle_protocol.shape[1]==3 and combinations["cycles"]== -1) | (cycle_protocol.shape[1]==12 and combinations["cycles"]==10):
                stat_res=topX(data_path, cycle_protocol, column=combinations["column_to_learn"])
                for i in range(0,len(stat_res)):
                    #print(i)
                    res[i]+= stat_res[i]
                stat_counter+=1


            else:
                print(hash + " ------ " + stat + " is invalid with the config:")
                print(combinations)
        if stat_counter != 3:
            print(stat_counter)
            print(hash)
            exit()



    for i in range(0,len(res)):
        res[i]= res[i]/avg_counter


    return res
    
    

def single_experiment_with_wiskers(combination):
    cycles=combination["cycles"]
    tuplist = ""
    if cycles == 10:
        res_num = 11
    else:
        res_num= 2

    for hash in os.listdir(FOLDER_PATH):
        combination_path = FOLDER_PATH+ hash + '/combination.txt'
        if not os.path.exists(combination_path):
            #print("no combination for hash", hash)
            continue



        combinations = {}
        with open(FOLDER_PATH+ hash + '/combination.txt', 'r') as file:
            combinations = json.loads(file.read())

        #selection logic

        if combination != combinations:
            continue


        data_path = combinations["ground_truth_path"]
        res = []
        for stat in os.listdir(FOLDER_PATH+ hash):
            if (stat == "combination.txt") | ( not os.path.exists(FOLDER_PATH+ hash + '/' + stat + "/cache.csv")):
                continue
            if stat == "7" or stat == "8" or stat == "9":
                continue
            cycle_protocol = pd.read_csv(FOLDER_PATH+ hash + '/' + stat+"/cache.csv")


            if (cycle_protocol.shape[1]==3 and combinations["cycles"]== -1) | (cycle_protocol.shape[1]==12 and combinations["cycles"]==10):
                pass
                res.append(topX(data_path, cycle_protocol, column=combinations["column_to_learn"]))
            else:
                print(hash + " ------ " + stat + " is invalid with the config:")
                print(combinations)

        #print(res)
        #print(hash)
        assert len(res)==3
        for i in range(0,res_num):

            mid = statistics.mean([res[0][i],res[1][i],res[2][i]])
            top = get_largest(res[0][i],res[1][i],res[2][i])
            bot = get_smallest(res[0][i],res[1][i],res[2][i])        

            tuplist+= str((i,mid)) +"+= (0,"+str(abs(mid-top))+")"
            tuplist+= "\n"
            tuplist+= str((i,mid)) +"-= (0,"+str(abs(mid-bot))+")"
            tuplist+= "\n"
            #thistuple = (get_smallest(res[0][i],res[1][i],res[2][i]), get_middle(res[0][i],res[1][i],res[2][i]), get_largest(res[0][i],res[1][i],res[2][i]))
            #ret.append(thistuple)
        return tuplist

def frame():
    for hash in os.listdir(FOLDER_PATH):
        combination_path = FOLDER_PATH+ hash + '/combination.txt'
        if not os.path.exists(combination_path):
            print("no combination for hash", hash)
            continue



        combinations = {}
        with open(FOLDER_PATH+ hash + '/combination.txt', 'r') as file:
            combinations = json.loads(file.read())

        #selection logic


        data_path = combinations["ground_truth_path"]
        res = []
        for stat in os.listdir(FOLDER_PATH+ hash):
            if (stat == "combination.txt") | ( not os.path.exists(FOLDER_PATH+ hash + '/' + stat + "/cache.csv")):
                continue
            
            cycle_protocol = pd.read_csv(FOLDER_PATH+ hash + '/' + stat+"/cache.csv")


            if (cycle_protocol.shape[1]==3 and combinations["cycles"]== -1) | (cycle_protocol.shape[1]==12 and combinations["cycles"]==10):
                pass
            else:
                print(hash + " ------ " + stat + " is invalid with the config:")
                print(combinations)

def find_unfinished012():
    for hash in os.listdir(FOLDER_PATH):
        combination_path = FOLDER_PATH+ hash + '/combination.txt'
        if not os.path.exists(combination_path):
            print("no combination for hash", hash)
            continue



        combinations = {}
        with open(FOLDER_PATH+ hash + '/combination.txt', 'r') as file:
            combinations = json.loads(file.read())
        #selection logic


        #data_path = combinations["ground_truth_path"]
        res = []
        count =0

        for stat in os.listdir(FOLDER_PATH+ hash):
            if (stat == "combination.txt"):
                continue
            if stat == "7" or stat == "8" or stat == "9":
                continue
            if ( not os.path.exists(FOLDER_PATH+ hash + '/' + stat + "/cache.csv")):
                print(hash + " ------ " + stat + " does not have a cache.csv")
                continue
            
            cycle_protocol = pd.read_csv(FOLDER_PATH+ hash + '/' + stat+"/cache.csv")


            if (cycle_protocol.shape[1]==3 and combinations["cycles"]== -1) | (cycle_protocol.shape[1]==12 and combinations["cycles"]==10):
                continue
            else:
                print(hash + " ------ " + stat + " is invalid with the config:")
                #print(combinations)



def cross_target_avg_rf_gredy_vs_random():
    for i in range(len(path_list)):
        _avg_of_all_greedy = {'learner': ['rf_learner'], 
                                 'hyperparameter_tuning': [False], 
                                 'batch_size_percentage': [1,0.5,0.1], 
                                 'smids_input_path': [path_list[i]], 
                                 'ground_truth_path': [path_list[i]], 
                                 'cycles': [10], #!ONLY ONE
                                 'column_to_learn': [''], 
                                 'do_scoring_function_list_prediction': [False], 
                                 'first_query_function': ['random_query_function'], 
                                 'query_function': ['greedy_query_function']}

        keys, values = zip(*_avg_of_all_greedy.items())
        avg_of_all_greedy = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print("avg_of_all_greedy:"+str(path_list[i]))
        coordinate_friendly_list_print(avg_combinatorial_space_no_wiskers(avg_of_all_greedy))
        print("\n")
        


    for i in range(len(path_list)):
        _avg_of_all_random = {'learner': ['rf_learner'], 
                                 'hyperparameter_tuning': [False], 
                                 'batch_size_percentage': [1,0.5,0.1], 
                                 'smids_input_path': [path_list[i]], 
                                 'ground_truth_path': [path_list[i]], 
                                 'cycles': [10], #!ONLY ONE
                                 'column_to_learn': [''], 
                                 'do_scoring_function_list_prediction': [False], 
                                 'first_query_function': ['random_query_function'], 
                                 'query_function': ['random_query_function']}

        keys, values = zip(*_avg_of_all_random.items())
        avg_of_all_random = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print("avg_of_all_random:"+str(path_list[i]))
        coordinate_friendly_list_print(avg_combinatorial_space_no_wiskers(avg_of_all_random))
        print("\n")


def cross_target_batch_size_comparison_greedy():
    for i in range(len(path_list)):
        for bs in [1,0.5,0.1,0.01]:
            _avg_of_all_greedy = {'learner': 'rf_learner', 
                                 'hyperparameter_tuning': False, 
                                 'batch_size_percentage': bs, 
                                 'smids_input_path': path_list[i], 
                                 'ground_truth_path': path_list[i], 
                                 'cycles': 10, #!ONLY ONE
                                 'column_to_learn': '', 
                                 'do_scoring_function_list_prediction': False, 
                                 'first_query_function': 'random_query_function', 
                                 'query_function': 'greedy_query_function'}

            print("batch_size_comparison:-----"+str(bs)+"----"+str(path_list[i]))
            print(single_experiment_with_wiskers(_avg_of_all_greedy))
    print("\n")
    #for i in len(path_list):


#comb= {"learner": "rf_learner", "hyperparameter_tuning": False, "batch_size_percentage": 1, "smids_input_path": "./data/GBA_2v3e_scoring_and_consensus_maxAL.csv", "ground_truth_path": "./data/GBA_2v3e_scoring_and_consensus_maxAL.csv", "cycles": 10, "column_to_learn": "ConvexPLR", "do_scoring_function_list_prediction": False, "first_query_function": "random_query_function", "query_function": "greedy_query_function"}
#print(single_experiment_with_wiskers(comb))

def tenvstwo():
    for i in range(len(path_list)):
        for bs in [1,0.1]:
            _avg_of_all_greedy = {'learner': 'rf_learner', 
                                 'hyperparameter_tuning': False, 
                                 'batch_size_percentage': bs, 
                                 'smids_input_path': path_list[i], 
                                 'ground_truth_path': path_list[i], 
                                 'cycles': 10, #!ONLY ONE
                                 'column_to_learn': '', 
                                 'do_scoring_function_list_prediction': False, 
                                 'first_query_function': 'random_query_function', 
                                 'query_function': 'greedy_query_function'}

            print("tenvstwo: cycle10 batch_size_comparison:-----"+str(bs)+"----"+str(path_list[i]))
            print(single_experiment_with_wiskers(_avg_of_all_greedy))
            print("\n")
            _avg_of_all_greedy = {'learner': 'rf_learner', 
                                 'hyperparameter_tuning': False, 
                                 'batch_size_percentage': bs, 
                                 'smids_input_path': path_list[i], 
                                 'ground_truth_path': path_list[i], 
                                 'cycles': -1, #!ONLY ONE
                                 'column_to_learn': '', 
                                 'do_scoring_function_list_prediction': False, 
                                 'first_query_function': 'random_query_function', 
                                 'query_function': 'greedy_query_function'}

            print("tenvstwo: cycle-1 batch_size_comparison:-----"+str(bs)+"----"+str(path_list[i]))
            print(single_experiment_with_wiskers(_avg_of_all_greedy))

def extensiveGBA():
    for bs in [1,0.5,0.1, 0.01]:
        for cy in [10,-1]:
            _avg_of_all_greedy = {'learner': 'rf_learner', 
                                 'hyperparameter_tuning': False, 
                                 'batch_size_percentage': bs, 
                                 'smids_input_path': path_list[1], 
                                 'ground_truth_path': path_list[1], 
                                 'cycles': cy, #!ONLY ONE
                                 'column_to_learn': '', 
                                 'do_scoring_function_list_prediction': False, 
                                 'first_query_function': 'random_query_function', 
                                 'query_function': 'greedy_query_function'}

            print(str("%")+"GBA-----"+str(bs)+"----"+str(cy)+"----rf_NOT_tuned")
            print(single_experiment_with_wiskers(_avg_of_all_greedy))
            print("\n")
            _avg_of_all_greedy = {'learner': 'rf_learner', 
                                 'hyperparameter_tuning': True, 
                                 'batch_size_percentage': bs, 
                                 'smids_input_path': path_list[1], 
                                 'ground_truth_path': path_list[1], 
                                 'cycles': cy, #!ONLY ONE
                                 'column_to_learn': '', 
                                 'do_scoring_function_list_prediction': False, 
                                 'first_query_function': 'random_query_function', 
                                 'query_function': 'greedy_query_function'}

            print(str("%")+"GBA-----"+str(bs)+"----"+str(cy)+"----rf_TUNED")
            print(single_experiment_with_wiskers(_avg_of_all_greedy))
            _avg_of_all_greedy = {'learner': 'cp_learner', 
                                 'hyperparameter_tuning': False, 
                                 'batch_size_percentage': bs, 
                                 'smids_input_path': path_list[1], 
                                 'ground_truth_path': path_list[1], 
                                 'cycles': cy, #!ONLY ONE
                                 'column_to_learn': '', 
                                 'do_scoring_function_list_prediction': False, 
                                 'first_query_function': 'random_query_function', 
                                 'query_function': 'greedy_query_function'}

            print(str("%")+"GBA-----"+str(bs)+"----"+str(cy)+"----chemprop")
            print(single_experiment_with_wiskers(_avg_of_all_greedy))







def main():
    #cross_target_avg_rf_gredy_vs_random()
    #cross_target_batch_size_comparison_greedy()
    #tenvstwo()
    extensiveGBA()
main()
