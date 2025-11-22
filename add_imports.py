#!/usr/bin/env python3
import sys

def add_imports(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Check if imports already exist
    has_scalaroperand = any('ScalarOperand' in line for line in lines)
    has_sum = any('use std::iter::Sum' in line for line in lines)

    # Find the last import line
    last_import_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('use '):
            last_import_idx = i

    if last_import_idx == -1:
        return  # No imports, skip

    # Add missing imports after the last import
    new_lines = []
    for i, line in enumerate(lines):
        new_lines.append(line)
        if i == last_import_idx:
            if not has_scalaroperand and 'ndarray' in ''.join(lines):
                # Check if ndarray is already imported
                for j in range(i+1):
                    if 'use ndarray::' in lines[j] or 'use ndarray::{' in lines[j]:
                        # Modify existing ndarray import
                        if 'ScalarOperand' not in lines[j]:
                            lines[j] = lines[j].rstrip().rstrip(';').rstrip('}')
                            if lines[j].endswith('{'):
                                lines[j] += 'ScalarOperand};\n'
                            else:
                                lines[j] += ', ScalarOperand};\n'
                        break
                else:
                    # No ndarray import found, add new one
                    new_lines.append('use ndarray::ScalarOperand;\n')

            if not has_sum:
                new_lines.append('use std::iter::Sum;\n')

    if new_lines != lines:
        with open(filename, 'w') as f:
            f.writelines(lines)  # Write original since we modified in place above

if __name__ == '__main__':
    add_imports(sys.argv[1])
