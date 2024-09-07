# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:33:15 2024

@author: Jonas
"""

#scripts = ['lmfit_vtraflux.py', 'vtraflux.py' ]
#scripts = ['lmfit_vtraflux.py' ]

#for script in scripts: 

#script_path = dname + '/' +  'lmfit_vtraflux.py'
script_path = dname + '/' + script# 'lmfit_vtraflux.py'

# Read the original script and split it into lines
with open(script_path, 'r') as f:
    script_lines = f.readlines()

# The line you want to add at the beginning
line_to_add = '@ray.remote(memory=250 * 1024 * 1024)\n'

# Specify the line number where you want to insert the new line
# Note: Line numbers start from 0 (0 = first line)
line_number = 8  # Insert the new line after the 2nd line (0-indexed)

if script == 'lmfit_vtraflux.py': line_number = 8
if script == 'vtraflux.py': 
    line_number = 10
    #print(line_number)
# Insert the new line at the specified position in the list
script_lines.insert(line_number, line_to_add)

#print(script_lines[1:20])


# Create a temporary file to save the modified script in text mode
with tempfile.NamedTemporaryFile(delete=False, suffix='.py', mode='w+') as temp_script:
    temp_script_name = temp_script.name
    # Write the modified content (list of lines) to the temporary file
    temp_script.writelines(script_lines)
    #print(script_lines)
# Import the modified script
spec = importlib.util.spec_from_file_location('modified_script', temp_script_name)
modified_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modified_script)


# Now you can use functions from the modified script, for example:
# result = modified_script.some_function()

# Clean up the temporary file
os.remove(temp_script_name)

