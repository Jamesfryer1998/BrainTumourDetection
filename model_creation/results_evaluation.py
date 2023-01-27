import os
import json

with open("model_creation/model_structure_testing.json", "r") as infile:
    data = json.load(infile)

sorted_list = sorted(data, key=lambda k: k['test_accuracy'], reverse=True) 

print(f'Best conv: {sorted_list[0]["num_conv"]}')
print(f'Best dense: {sorted_list[0]["num_dense"]}')
print(f'Test accuracy: {sorted_list[0]["test_accuracy"]:.3f}')
print(f'Test loss: {sorted_list[0]["test_loss"]:.3f}')