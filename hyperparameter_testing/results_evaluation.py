import json
from datetime import datetime, timedelta

def evaluate_hyperparameter_optimisation(print_results=False):
    
    with open("hyperparameter_testing/hyperparamater_results.json", "r") as infile:
        data = json.load(infile)

    print(f'\nCombinatatiuons configured {len(data)}\n')

    sorted_list = sorted(data, key=lambda k: k['test_accuracy'], reverse=True) 

    if print_results == True:
        print('--Evaluation of Hyperparameter Optimisation--')
        print(f'time: {sorted_list[0]["time"]}')
        print(f'num_conv_layers: {sorted_list[0]["num_conv_layers"]}')
        print(f'num_dense_layers: {sorted_list[0]["num_dense_layers"]}')
        print(f'conv_1_2_unit: {sorted_list[0]["conv_1_2_unit"]}')
        print(f'conv_3_4_unit: {sorted_list[0]["conv_3_4_unit"]}')
        print(f'dense_unit: {sorted_list[0]["dense_unit"]}')
        print(f'epoch: {sorted_list[0]["epoch"]}')
        print(f'test_loss: {sorted_list[0]["test_loss"]:.3f}')
        print(f'test_accuracy: {sorted_list[0]["test_accuracy"]:.3f}\n')

        print('--Improvement of Hyperparameter Optimisation--')
        print(f'Best test_accuracy: {sorted_list[0]["test_accuracy"]:.3f}')
        print(f'Worst test_accuracy: {sorted_list[-1]["test_accuracy"]:.3f}')
        print(f'Improvement in test_accuracy: {sorted_list[0]["test_accuracy"]- sorted_list[-1]["test_accuracy"]:.3f}\n')

        #Sum up time taken to complete everything
        def sum_times(data):
            total_time = timedelta()
            time_list = [i['time'] for i in data]
            for time_dict in time_list:
                time = datetime.strptime(time_dict, "%H:%M:%S.%f")
                total_time += timedelta(hours=time.hour, minutes=time.minute, seconds=time.second)
            return total_time

        print(f'Hyperparamater optimisation took: {sum_times(data)}')

        return sorted_list[0]["conv_1_2_unit"], sorted_list[0]["conv_3_4_unit"], sorted_list[0]["dense_unit"], sorted_list[0]["epoch"]