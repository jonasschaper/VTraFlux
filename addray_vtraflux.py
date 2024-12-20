# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:33:15 2024

@author: Jonas
"""


script_path = dname + '/' + script

with open(script_path, 'r') as f:
    script_lines = f.readlines()

line_to_add = '@ray.remote(memory=250 * 1024 * 1024)\n'


line_number = 8  

if script == 'lmfit_vtraflux.py': line_number = 8
if script == 'vtraflux.py': 
    line_number = 10
script_lines.insert(line_number, line_to_add)


with tempfile.NamedTemporaryFile(delete=False, suffix='.py', mode='w+') as temp_script:
    temp_script_name = temp_script.name
    temp_script.writelines(script_lines)

spec = importlib.util.spec_from_file_location('modified_script', temp_script_name)
modified_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modified_script)

os.remove(temp_script_name)

