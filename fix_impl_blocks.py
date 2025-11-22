#!/usr/bin/env python3
import re
import sys

def fix_impl_blocks(content):
    """Fix malformed impl blocks with where clauses."""
    # Pattern matches:
    # impl<A>
    # where
    #     A: Float + ScalarOperand + Sum, TYPENAME {
    pattern = r'impl<A>\nwhere\n    A: Float \+ ScalarOperand \+ Sum, (.*) \{'
    replacement = r'impl<A> \1\nwhere\n    A: Float + ScalarOperand + Sum,\n{'

    return re.sub(pattern, replacement, content)

if __name__ == '__main__':
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        content = f.read()

    fixed = fix_impl_blocks(content)

    with open(filename, 'w') as f:
        f.write(fixed)

    print(f"Fixed {filename}")
